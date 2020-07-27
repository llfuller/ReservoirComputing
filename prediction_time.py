import numpy as np
import Grid_Search_Settings
import time
import Lorenz96
import Lorenz63
import Colpitts
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import Plotting

"""
Discover at which time predictions error crosses a threshold and save the results it.
"""


def time_exceeds_threshold(anArray, threshold_value):
    """
    Returns index at which threshold value is exceeded by element of monotonically increasing anArray
    """
    return_index = 0
    for i, aNum in enumerate(anArray):
        if aNum>=threshold_value:
            return i

    # At this point no returned value (threshold never exceeded), so just return max index in anArray
    return np.shape(anArray)[0]-1

def run_This_Method():
    threshold = 0.1
    system_name = "Colpitts"
    perform_grid_search = True
    run_system = False
    setup_number = 2
    train_start_timestep = 0
    train_end_timestep = 6000 # Timestep at which training ends.
    timesteps_for_prediction = 1000 # if this is too big, then calculations take too long. Medium range: MSE is meaningless. Short range: good measure
    # what the overall prediction behavior is.
    noise_std_dev = 0.00

    start_time = time.time()

    #=======================================================================================================================
    # Data Input
    #=======================================================================================================================
    if system_name is "L63":
        if run_system:
            Lorenz63.run_L63(t_final = 20000.0,
                             dt = 0.002)
        N_u, N_y = 3, 3

    if system_name is "L96":
        dims = 5 # I've only seen this work well up to 5 dimensions
        if run_system:
            Lorenz96.run_L96(dims,
                             t_final = 2000.0,
                             dt = 0.001)
        N_u, N_y = dims, dims

    if system_name is "Colpitts":
        if run_system:
            Colpitts.run_Colpitts(t_final = 10000.0,
                                  dt = 0.01)
        N_u, N_y = 3, 3


    #copy of imported file which only uses 1 out of every 10 timesteps:
    state_target = (   (np.loadtxt(system_name+'_states.txt')[::10]).transpose()   ).copy()
    print("Shape of state_target array: "+str(np.shape(state_target)))
    num_timesteps_data = np.shape(state_target)[1]

    state_target = np.divide(state_target,np.max(np.abs(state_target))) # Actual normalized input to reservoir.
    print("Finished loading and processing target state.")
    #=======================================================================================================================
    # Grid Search Lists
    #=======================================================================================================================

    # Grid search lists:
    list_of_scaling_W, list_of_scaling_alpha, list_of_beta_to_test, list_of_scaling_W_fb, \
    list_of_scaling_W_in = Grid_Search_Settings.Set_Grid(state_target, perform_grid_search, setup_number)
    print("shape of W_fb: " +str(np.shape(list_of_scaling_W_fb)[0]))
    length_of_W_list = np.shape(list_of_scaling_W)[0]
    length_of_W_in_list = np.shape(list_of_scaling_W_in)[0]
    length_of_scaling_alpha_list = np.shape(list_of_scaling_alpha)[0]
    length_of_beta_to_test_list = np.shape(list_of_beta_to_test)[0]
    length_of_scaling_W_fb_list = np.shape(list_of_scaling_W_fb)[0]
    print("Finished loading grid parameters.")

    #=======================================================================================================================
    # Defining Error
    #=======================================================================================================================

    # error at time t
    current_errors = np.zeros((length_of_W_list,
                               length_of_W_in_list * length_of_scaling_alpha_list * length_of_beta_to_test_list * \
                               length_of_scaling_W_fb_list,
                               train_end_timestep+timesteps_for_prediction
                               ))

    # cumulative error by time t
    cumulative_errors = np.zeros((length_of_W_list,
                               length_of_W_in_list * length_of_scaling_alpha_list * length_of_beta_to_test_list * \
                               length_of_scaling_W_fb_list,
                               train_end_timestep+timesteps_for_prediction
                               ))

    #=======================================================================================================================
    # Calculating Errors
    #=======================================================================================================================
    print("Start calculating errors: ")
    state_target_transposed = state_target.transpose()
    threshold_pass_time_array =  np.empty((length_of_W_list,
                               length_of_W_in_list * length_of_scaling_alpha_list * length_of_beta_to_test_list * \
                               length_of_scaling_W_fb_list
                               ))
    for j, scaling_W in enumerate(list_of_scaling_W):
        print("scaling_W = " + str(scaling_W))
        index2 = 0
        for i, scaling_W_in in enumerate(list_of_scaling_W_in):
            for k, scaling_alpha in enumerate(list_of_scaling_alpha):
                for l, beta in enumerate(list_of_beta_to_test):
                    for m, scaling_W_fb in enumerate(list_of_scaling_W_fb):
                        print("index2="+str(index2))
                        index1 = j
                        print("Index2 is "+str(index2))
                        params = [scaling_W_in,
                                  scaling_W,
                                  scaling_alpha,
                                  beta,
                                  scaling_W_fb]
                        state = np.loadtxt("states/" + system_name + "/" +
                                           "prediction/"+
                                           "_orbit_params_(" +
                                           str(round(params[0], 4)) + "," +
                                           str(round(params[1], 4)) + "," +
                                           str(round(params[2], 4)) + "," +
                                           "{:.2e}".format(params[3])+ ","+
                                           str(round(params[4], 4)) + ").txt")
                        # print("Shape of state array: " + str(state))
                        # For each timestep:
                        for n in range(train_end_timestep+1, train_end_timestep+timesteps_for_prediction):
                            current_errors[index1, index2, n] = mean_squared_error(
                                state_target_transposed[train_end_timestep:n],
                                state.transpose()[train_end_timestep:n])
                            # print("current error: " +str(current_errors[index1, index2, n]))
                            cumulative_errors[index1, index2, n-1] = np.sum(current_errors[index1, index2, train_end_timestep:n])
                            print(str(cumulative_errors[index1, index2, n-2])+"+"+str(current_errors[index1, index2, n-1])+"="+str(cumulative_errors[index1, index2, n-1])+"=?="+str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(cumulative_errors[index1, index2, n-1] + current_errors[index1, index2, n]))
                            print(str(np.sum(current_errors[index1, index2, train_end_timestep:n])))
                            print(str(np.sum(current_errors[index1, index2, train_end_timestep:n])))
                            print(str(np.sum(current_errors[index1, index2, train_end_timestep:n])))
                            print(str(np.sum(current_errors[index1, index2, train_end_timestep:n])))
                            # print("From before: " + np.sum(current_errors[index1, index2, train_end_timestep:n]))
                        plt.figure()
                        times_plotted = np.arange(0, train_end_timestep+timesteps_for_prediction)
                        cumulative_errors_plotted = cumulative_errors[index1, index2]
                        # print("pltx: \n"+str(pltx))
                        # print("plty: \n"+str(plty))
                        plt.plot(times_plotted, cumulative_errors_plotted)
                        plt.xlim(train_end_timestep, train_end_timestep+timesteps_for_prediction)
                        plt.plot(times_plotted, threshold*np.ones(train_end_timestep+timesteps_for_prediction), 'r:')
                        plt.xlabel("Timestep")
                        plt.ylabel("Cumulative Error")
                        plt.savefig("cumulative_error/plots/"+system_name+
                                           "cumulative_error_orbit_params_(" +
                                           str(round(params[0], 4)) + "," +
                                           str(round(params[1], 4)) + "," +
                                           str(round(params[2], 4)) + "," +
                                           "{:.2e}".format(params[3])+ ","+
                                           str(round(params[4], 4)) + ").png")
                        np.savetxt("cumulative_error/data/"+system_name+
                                           "cumulative_error_orbit_params_(" +
                                           str(round(params[0], 4)) + "," +
                                           str(round(params[1], 4)) + "," +
                                           str(round(params[2], 4)) + "," +
                                           "{:.2e}".format(params[3])+ ","+
                                           str(round(params[4], 4)) + ").txt", cumulative_errors_plotted)
                        print("SAVING: cumulative_error/plots/"+
                                           "cumulative_error_orbit_params_(" +
                                           str(round(params[0], 4)) + "," +
                                           str(round(params[1], 4)) + "," +
                                           str(round(params[2], 4)) + "," +
                                           "{:.2e}".format(params[3])+ ","+
                                           str(round(params[4], 4)) + ").png")
                        threshold_pass_time = time_exceeds_threshold(cumulative_errors_plotted, threshold)
                        threshold_pass_time_array[index1, index2] = threshold_pass_time
                        print("Threshold exceeded at "+str(threshold_pass_time))
                        print("Value before threshold: "+str(cumulative_errors_plotted[threshold_pass_time-1]))
                        print("Value after threshold: "+str(cumulative_errors_plotted[threshold_pass_time]))
                        print("---------------")
                        # Plotting.plot_orbits(state, state_target, train_start_timestep, train_end_timestep, system_name,
                        #                      timesteps_for_prediction, setup_number,
                        #                      params, "2d display")
                        # plt.show()
                        index2+=1

    np.savetxt("threshold_pass_time_array_"+system_name+"_setup_"+str(setup_number)+".txt",threshold_pass_time_array)

    #=======================================================================================================================
    # Plotting Spectral Radius vs Prediction Time
    #=======================================================================================================================
    threshold_pass_time_array_loaded = np.loadtxt("threshold_pass_time_array_"+system_name+"_setup_"+str(setup_number)+".txt")
    print(np.shape(threshold_pass_time_array_loaded))
    averaged_prediction_time = np.average(threshold_pass_time_array_loaded, axis=1)
    print(np.shape(averaged_prediction_time))
    print(averaged_prediction_time)
    plt.figure()
    plt.plot(list_of_scaling_W, averaged_prediction_time)
    plt.xlabel("Spectral Radius")
    plt.ylabel("Prediction Time (threshold = 0.1)")
    plt.title("Spectral Radius Average Performance for "+system_name+" with Grid Setup "+str(setup_number))
    plt.savefig("Spectral Radius Average Performance for "+system_name+" with Grid Setup "+str(setup_number)+".png")
    # plt.show()