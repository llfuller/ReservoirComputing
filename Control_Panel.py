import Lorenz63
import Lorenz96
import Colpitts
import Grid_Search_Settings
import ESN_Process
import ESN
import numpy as np
import time
import os.path
from sklearn.metrics import mean_squared_error
from scipy import stats
import Plotting
import prediction_time

"""
This script is the only script to run. It uses the other scripts in the directory to run sub-tasks.
Parameters are specified at the top.
This is used to create graphs of outputs for single ESNs and grid search.
"""
system_name_list = ["L96"]
for system_name in system_name_list:
    for dims in [4,5,6,7,8,9,10,11,12]:
        print("******************************************************")
        # system_name = "Colpitts"
        setup_number = 2

        # for system_name in ["L96"]:
        #     for setup_number in [1]:
        #         for dims in [5,6,8,9,36]:
        print("Running for "+str(dims)+" dims")
        np.random.seed(2020)
        #=======================================================================================================================
        # Run Parameters
        #=======================================================================================================================
        run_system = True # Generate new data from chosen system. If L96, should run this system if changing dimensions.
        N_x = 3000 # Number of nodes in reservoir."should be at least equal to the estimate of independent real values
        # the reservoir has to remember from the input to solve its task"
        # -Lukosevicius in PracticalESN
        perform_grid_search = False
        # setup_number = -1
        sparsity_tuples = np.array([[0.06*N_x/N_x,1.0]
                                    ])
        produce_data = True
        preload_W = True
        # First value: sparsity (numerator is average number of connections FROM one node TO other nodes),
        # second value: proportion of network with that sparsity
        # sparsity = 15.0 / N_x # Only applies to GPU so far TODO: What if symmetric?
        train_start_timestep = 0
        train_end_timestep = 3000 # Timestep at which training ends.
        timesteps_for_prediction = 1000 # if this is too big, then calculations take too long. Medium range: MSE is meaningless. Short range: good measure
        # what the overall prediction behavior is.
        save_or_display = "2d None"  #save 3d or 3d plots of orbits after prediction or display them. Set to None for neither.
        # use 3d or 2d prefix for either type of graph.
        print_timings_boolean = True
        # Since all data is normalized, the characteristic length is 1. I'll set the allowed deviation length to 0.05 of this.
        save_name = "ESN_1"
        dev_length_multiplier = 2.0
        noise_std_dev = 0.00

        start_time = time.time()

        #=======================================================================================================================
        # Data Input
        #=======================================================================================================================
        if system_name is "L63":
            dims = 3
            if run_system:
                Lorenz63.run_L63(t_final = 20000.0,
                                 dt = 0.002)
            N_u, N_y = 3, 3

        if system_name is "L96":
            # dims = 10  # dims for L96 I've only seen this work well up to 5 dimensions
            print("dims "+str(dims))
            print(run_system)
            if run_system:
                Lorenz96.run_L96(dims,
                                 t_final = 2000.0,
                                 dt = 0.001)
            N_u, N_y = dims, dims

        if system_name is "Colpitts":
            dims = 3
            if run_system:
                Colpitts.run_Colpitts(t_final = 10000.0,
                                      dt = 0.01)
            N_u, N_y = 3, 3

        #=======================================================================================================================
        # Directory and Saving Info
        #=======================================================================================================================

        dimension_directory = ""  # set like this so it's blank if not L96
        if system_name.upper() == "L96":
            dimension_directory = str(dims) + "D/"

        #copy of imported file which only uses 1 out of every 10 timesteps:
        state_target = (   (np.loadtxt(system_name+'_states.txt')[::10]).transpose()   ).copy()
        print("Shape of state_target array: "+str(np.shape(state_target)))
        num_timesteps_data = np.shape(state_target)[1]

        state_target = np.divide(state_target,np.max(np.abs(state_target))) # Actual normalized input to reservoir.
        noise_array = np.random.normal(loc = 1.0, scale = noise_std_dev, size = np.shape(state_target))
        state_target_noisy = np.multiply(noise_array,state_target)
        if not os.path.isfile("states/" + system_name + "/" + "target_noisy/"+dimension_directory+system_name+"target_noisy_"+str(noise_std_dev)+".txt"):
            np.savetxt("states/" + system_name + "/" + "target_noisy/"+dimension_directory+system_name+"target_noisy_"+str(noise_std_dev)+".txt",
                       state_target_noisy[:, :train_end_timestep + timesteps_for_prediction])
        if not os.path.isfile("states/" + system_name + "/" + "target_no_noise/" +dimension_directory+ system_name + "target.txt"):
            np.savetxt("states/" + system_name + "/" + "target_no_noise/" +dimension_directory+ system_name + "target.txt",
                       state_target[:, :train_end_timestep + timesteps_for_prediction])

        print("Max of target array")
        print(np.max(np.abs(state_target)))
        state = np.empty((N_y, train_end_timestep+timesteps_for_prediction)) # Input to reservoir. Before train_end_timestep,
        # state is identical to state_target. After that index, it will differ as this is a prediction of the state by the
        # reservoir.
        # Lorenz96.plot_L96()
        #=======================================================================================================================
        # Grid Search Section
        #=======================================================================================================================
        i,j,k,l,m = 0,0,0,0,0 # Indices to use in each grid if not grid searching
        list_of_scaling_W, list_of_scaling_alpha, list_of_beta_to_test, list_of_scaling_W_fb, \
        list_of_scaling_W_in = Grid_Search_Settings.Set_Grid(state_target, perform_grid_search, setup_number)
        # These values are used if no looping over that list:
        scaling_W = list_of_scaling_W[0]
        scaling_W_in = list_of_scaling_W_in[0]
        scaling_W_fb = list_of_scaling_W_fb[0]
        scaling_alpha = list_of_scaling_alpha[0]

        # Keeping track of mse:
        mse_array = 100000000*np.ones((np.shape(list_of_scaling_W_in)[0],
                              np.shape(list_of_scaling_W)[0],
                              np.shape(list_of_scaling_alpha)[0],
                              np.shape(list_of_beta_to_test)[0],
                                np.shape(list_of_beta_to_test)[0]))

        #=======================================================================================================================
        # Construct, Train, and Predict
        #=======================================================================================================================
        x_initial = np.zeros(N_x)
        alpha_scatter_array_before_scaling = np.random.uniform(low=1,high=1,size=N_x)#np.random.normal(loc=1.0,scale=alpha_sigma,size=N_x)

        # Construct ESN (parameters will be reset in the loop later on)
        print("Now building ESN at time " + str(time.time() - start_time))
        placeholder_array = np.ones(1) # will be replaced later
        # if N_x <= 6000:
        ESN_Build_Method = ESN.ESN_CPU
        print("Using CPU")
        # else:
        #     ESN_Build_Method = ESN.ESN_GPU
        #     print("Using GPU")

        ESN_1 = ESN_Build_Method(N_x, N_u, N_y, sparsity_tuples,
                                 x_initial, scaling_alpha * alpha_scatter_array_before_scaling, scaling_W,
                                 scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction)
        ESN_list = [ESN_1]
        # for i in range(10):
        #     ESN_list.append(ESN_Build_Method(N_x, N_u, N_y, sparsity_tuples,
        #                          x_initial, float(i)/50*scaling_alpha * np.ones(N_x), scaling_W,
        #                          scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction))
        Group_1 = ESN.Reservoir_Group(ESN_list)


        if preload_W:
            sparsity_tuples_list = tuple([(a_row[0], a_row[1]) for a_row in sparsity_tuples])
            print("PreLoading W")
            # If file already exists
            preloaded_W = np.loadtxt('./W_(adjacency)/W_' + str(N_x) + '_' + str(N_x) + '_' + str(sparsity_tuples_list) + '.txt')
        else:
            preloaded_W = None

        print("Done building/loading ESN at time " + str(time.time() - start_time))
        # Training and Prediction
        #[scaling_W_in,
         # scaling_W,
         # scaling_alpha,
         # beta,
         # scaling_W_fb],

        # Beginning of edits for testing generalized synchronization #
        pnz_array = np.linspace(0.002,0.15,5)
        sr_array = np.linspace(0.002,10,20)
        random_initial_condition_1 = np.random.random((N_x))
        random_initial_condition_2 = np.random.random((N_x))
        random_initial_condition_list = [random_initial_condition_1, random_initial_condition_2]
        mse_array = 1000*np.ones((np.shape(pnz_array)[0], np.shape(sr_array)[0]))
        gen_sync_result_array = 1000*np.ones((np.shape(pnz_array)[0], np.shape(sr_array)[0]))
        print(np.shape(mse_array))
        for p, pnz in enumerate(pnz_array):
            sparsity_tuples = np.array([[pnz, 1.0]
                                        ])
            list_of_scaling_W = sr_array
            # End of edits for testing generalized synchronization #
            for i, scaling_W_in in enumerate(list_of_scaling_W_in):
                for j, scaling_W in enumerate(list_of_scaling_W):
                    for k, scaling_alpha in enumerate(list_of_scaling_alpha):
                        for m, scaling_W_fb in enumerate(list_of_scaling_W_fb):
                            # The beta loop is located inside ESN_Process because that is more efficient
                            print("------------------\n")
                            res_states_list = [-1,-1]
                            params = [scaling_W_in,
                                      scaling_W,
                                      scaling_alpha,
                                      list_of_beta_to_test[0],
                                      scaling_W_fb]
                            if produce_data:
                                ESN_Process.build_and_train_and_predict(Group_1, perform_grid_search,
                                                                        start_time, train_start_timestep, train_end_timestep,
                                                                        mse_array, list_of_beta_to_test, N_u, N_y, N_x, x_initial,
                                                                        state_target, state_target_noisy,
                                                                        scaling_W_fb, timesteps_for_prediction, scaling_W_in,
                                                                        system_name, dims, dimension_directory,
                                                                        print_timings_boolean, scaling_alpha,
                                                                        scaling_W, save_or_display,
                                                                        state, save_name, sparsity_tuples, preload_W, preloaded_W,
                                                                        alpha_scatter_array_before_scaling, setup_number,
                                                                        random_initial_condition_list, pnz, res_states_list,
                                                                        param_array=[i,j,k,m])
                                states_1 = res_states_list[0]
                                states_2 = res_states_list[1]
                            else: # load data instead
                                states_1 = np.loadtxt("states/" + system_name + "/" +
                                       "prediction_gen_synch/index_1/"+dimension_directory+
                                       "_orbit_params_(" +
                                       str(round(params[0], 4)) + "," +
                                       str(round(params[1], 4)) + "," +
                                       str(round(params[2], 4)) + "," +
                                       "{:.2e}".format(params[3])+ ","+
                                       str(round(params[4], 4)) + ")_pnz="+str(pnz)+".txt")
                                states_2 = np.loadtxt("states/" + system_name + "/" +
                                       "prediction_gen_synch/index_2/"+dimension_directory+
                                       "_orbit_params_(" +
                                       str(round(params[0], 4)) + "," +
                                       str(round(params[1], 4)) + "," +
                                       str(round(params[2], 4)) + "," +
                                       "{:.2e}".format(params[3])+ ","+
                                       str(round(params[4], 4)) + ")_pnz="+str(pnz)+".txt")
                                print(pnz)
                                print(str(round(params[1], 4)))
                                print("States values")
                                print(states_1[0,0])
                                print(states_2[0,0])

                                pass
                            mse_array[p, j] = mean_squared_error(states_1, states_2)
                            slope, intercept, r_value, p_value, std_error = stats.linregress(states_1.flatten(),
                                                                                             states_2.flatten())  # put all times into one row before performing linear regression
                            print("Slope, which should = 1: " + str(slope))
                            print("Standard error, which is hopefully small:" + str(std_error))
                            gen_sync_result_array[p, j] = (std_error < 0.0001)

                            # mse_array = np.loadtxt('mse_array_gen_sync_'+system_name)
                            # mse_array[0,0]=-5
                            # mse_array[-1,-1]=5
                            # mse_array[0,-1] = 10
                            # gen_sync_result_array[0,0]=-5
                            # gen_sync_result_array[-1,-1]=100
                            # gen_sync_result_array[0,-1] = 200

                            Plotting.plot_scatter(states_1[100:], states_2[100:],
                                       system_name+"_orbit_params_(" +
                                       str(round(params[0], 4)) + "," +
                                       str(round(params[1], 4)) + "," +
                                       str(round(params[2], 4)) + "," +
                                       "{:.2e}".format(params[3])+ ","+
                                       str(round(params[4], 4)) + ")_pnz="+str(pnz),
                                                  "Gen_Sync_Plots/Relationship_Charts/"+system_name+"/"+dimension_directory,
                                                  x_label="Initial Condition 1", y_label="Initial Condition 2",)
        Plotting.plot_contour(pnz_array, sr_array, gen_sync_result_array.transpose(), x_label='pnz',
                              y_label='SR',
                              title='Gen Sync for ' + str(
                                  system_name)+ " " + str(dims) + "-D" + " Driving of Tanh Neurons for 2 Different ICs")
        if system_name == "L96":
            np.savetxt('mse_array_gen_sync_'+str(system_name)+" "+str(dims)+"-D", mse_array)
        else:
            np.savetxt('mse_array_gen_sync_' + str(system_name), mse_array)
        # Plotting.plot_sr_pnz_gen_synch_heatmap(mse_array, pnz_array, sr_array)
        print("MSE for Lowest pnz and lowest SR:")
        print(mse_array[0,0])
        print("MSE for Largest pnz and largest SR:")
        print(mse_array[-1,-1])
        print("MSE for Lowest pnz and largest SR:")
        print(mse_array[0,-1])
        Plotting.plot_contour(pnz_array, sr_array, mse_array.transpose(), x_label='pnz', y_label='SR',
                              title='More SR MSE for '+str(system_name)+ " " + str(dims) + "-D" + " Driving of Tanh Neurons for 2 Different Initial Conditions")

        # Plotting.plot_1D_quick(states_1[:500,:4])
        # Plotting.plot_1D_quick(states_2[:500,:4])
        print("States values")
        print(states_1[0, 0])
        print(states_2[0, 0])

        # print("Minimum MSE of " +str(mse_array.min()))
        # indices_of_min = np.unravel_index(mse_array.argmin(), mse_array.shape)
        # print("Min MSE at parameter indices: "+str(indices_of_min))
        # print("Min MSE at parameters: "+"("+str(list_of_scaling_W_in[indices_of_min[0]])+","+
        #       str(list_of_scaling_W[indices_of_min[1]])+","+
        #       str(list_of_scaling_alpha[indices_of_min[2]])+","+
        #       str(list_of_beta_to_test[indices_of_min[3]])+","+
        #       str(list_of_scaling_W_fb[indices_of_min[4]])+")")
        # # Saving MSE matrix for later retrieval
        # if perform_grid_search:
        #     np.savez("mse_array_"+system_name+"_N_x_"+str(N_x)+"_setup"+str(setup_number)+"_Train"+str(train_end_timestep)+"_Predict_"+str(timesteps_for_prediction)+".npz",mse_array=mse_array)
        # else:
        #     np.savez("mse_array_"+system_name+"_N_x_"+str(N_x)+"_setupCustom"+"_Train"+str(train_end_timestep)+"_Predict_"+str(timesteps_for_prediction)+".npz",mse_array=mse_array)

        print("Done at time: "+str(time.time()-start_time))
        print("Plotting Lorenz Map for "+str(system_name))
        # prediction_time.run_This_Method()