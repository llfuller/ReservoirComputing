import time
import numpy as np
import Plotting
from sklearn.metrics import mean_squared_error
import scipy as sp
import matplotlib.pyplot as plt

def print_timing(print_timings_boolean, start_time, variable1_str):
    if print_timings_boolean==True:
        print(variable1_str+str(time.time()-start_time)+"\n-----------------------")


def build_and_train_and_predict(Group_obj, perform_grid_search, start_time,train_start_timestep,train_end_timestep,mse_array,
                                list_of_beta_to_test, N_u,N_y,N_x,x_initial,state_target, state_target_noisy,
                                scaling_W_fb, timesteps_for_prediction, scaling_W_in, system_name, dims, dimension_directory,
                                print_timings_boolean, scaling_alpha, scaling_W, save_or_display, state, save_name,
                                sparsity_tuples, preload_W, preloaded_W, alpha_scatter_array_before_scaling, time_sequence,
                                setup_number, extra_stuff_list, data_storage_number, param_array):
    if system_name == "NaKL":
        I_L63 = extra_stuff_list[0]
    # First thing: Set parameters of ESN_obj correctly for this run:
    for ESN_obj in Group_obj.list_of_ESN_objs:
        ESN_obj.W_in = ESN_obj.build_W_in(N_x, N_u, scaling_W_in)
        if preload_W: # just multiply preloaded W by scaling factor
            ESN_obj.W = np.multiply(scaling_W, preloaded_W)
        else: # have to build or load W from scratch everytime
            print("Building W matrix")
            ESN_obj.W = ESN_obj.build_W(N_x, sparsity_tuples, scaling_W)
        ESN_obj.alpha_matrix = ESN_obj.build_alpha_matrix(scaling_alpha * alpha_scatter_array_before_scaling)
        ESN_obj.W_fb = ESN_obj.build_W_fb(N_x, N_u, scaling_W_fb)
        spectral_radius = sp.amax(abs(sp.linalg.eigvals(ESN_obj.W)))
        print("scaling_W is " + str(scaling_W))
        print("Spectral radius is now: "+str(spectral_radius))
    # Create "echoes" and record the activations
    # Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
    for n in range(0, train_end_timestep):
        # Using Teacher Forcing method:
        #ESN_obj.update_reservoir(state_target[:, n], n, state_target[:,n+1])
        Group_obj.update_reservoirs(state_target_noisy[:, n], n, state_target_noisy[:,n+1])

    # time.sleep(1) # necessary to prevent CPU method from messing up calculation (allows last x update to complete?)
    print_timing(print_timings_boolean, start_time, "after_ESN_feed_time")


    # m,n,h boundary function
    def bound_variables(state, var_index, n):
        if state[var_index, n + 1] > 1:
            state[var_index, n + 1] = 1
        elif state[var_index, n + 1] < 0:
            state[var_index, n + 1] = 0
    for l, beta in enumerate(list_of_beta_to_test):
        i, j, k, m = param_array
        print("Testing for " + str((scaling_W_in, scaling_W, scaling_alpha, beta, scaling_W_fb)))
        print("Has Indices: " + str((i, j, k, l, m)))

        # Compute W_out (readout coefficients)
        Group_obj.calculate_W_out(state_target_noisy, beta, train_start_timestep, train_end_timestep)
        print_timing(print_timings_boolean, start_time, "after_W_out_train_time")

        # Clear activations and state from train_end_timestep onward to make sure they aren't
        # being used improperly during prediction:
        for ESN_obj in Group_obj.list_of_ESN_objs:
            ESN_obj.x[train_end_timestep+1:] = 0
        state[:,train_end_timestep+1:] = 0
        # Make prediction before end of training identical to target state
        state[:, 2:train_end_timestep+1] = state_target_noisy[:, 2:train_end_timestep+1]
        #If NaKL Model, make sure current (5th component of state) is clamped to true current at each timestep
        if system_name == "NaKL":
            state[4, :] = state_target[4, :np.shape(state)[1]]
            print(state[4])
            print(I_L63)
        # Predict state at each next timestep, keeping training progress:
        for n in range(train_end_timestep, train_end_timestep + timesteps_for_prediction-1):
            state[:, n+1] = Group_obj.output_Y(state[:, n], n)
            if system_name == "NaKL":
                state[4, n+1] = state_target[4, n+1] # this clamps current to correct value during prediction
                if np.abs(np.max(state[:, n + 1])) > 100:
                    print("This prediction must be ended (numbers getting too large)")
                    break
                for var_index in [1,2,3]:
                    bound_variables(state, var_index, n)

            Group_obj.update_reservoirs(state[:, n], n, state[:,n+1]) # generates x_(n+1)
        print_timing(print_timings_boolean, start_time, "after_Y_predict_train_time")

        # Determine output of reservoir using reservoir activity before training and W_out from training
        state_pre_training = np.empty((5, train_end_timestep))
        state_pre_training[:,0] = np.zeros(np.shape(state[:,0])) #set initial condition
        for n in range(0, train_end_timestep-1):
            state_pre_training[:, n+1] = Group_obj.output_Y(state[:, n], n)

        np.savetxt("NaKL+L63x_Prediction_20000_to_30000.txt", state[:,train_end_timestep:train_end_timestep+timesteps_for_prediction])
        np.savetxt("NaKL+L63x_Training_0_to_20000.txt", np.hstack((state_pre_training[:,0:train_end_timestep].transpose(),
                                                                           np.array([(time_sequence)[0:train_end_timestep]  ]).transpose(),
                                                                           state_target[:,0:train_end_timestep].transpose()) ).round(decimals=8), fmt='%.7f')
        np.savetxt("NaKL+L63x_Prediction_20000_to_30000_2.txt", np.hstack((state[:,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(),
                                                                           np.array([(time_sequence)[train_end_timestep:train_end_timestep+timesteps_for_prediction]  ]).transpose(),
                                                                           state_target[:,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose()) ).round(decimals=8), fmt='%.7f')
        np.savetxt("NaKL+L63x_Prediction_20000_to_30000_dt_0.01_mhn_only.txt", np.hstack((state[1:4,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(),
                                                                           np.array([(time_sequence)[train_end_timestep:train_end_timestep+timesteps_for_prediction]  ]).transpose(),
                                                                           state_target[1:4,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose()) ).round(decimals=8), fmt='%.7f')
        # plt.figure()
        # plt.plot(state_target[:,0:train_end_timestep].transpose())
        # plt.plot(state_pre_training[:,0:train_end_timestep].transpose(), linestyle = ":")
        # plt.show()
        # plt.figure()
        # plt.plot(state_target[:,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())
        # plt.plot(state[:,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(), linestyle = ":")
        # plt.show()

        # np.savez(save_name+'.npz', ESN_obj=ESN_obj)
        # print("mean_squared_error is: " + str(mean_squared_error(
        #     state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
        #     state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())))
        # mse_array[i, j, k, l] = mean_squared_error(
        #     state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
        #     state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())
        # print("Number of large W_out:"+str(np.sum(ESN_obj.W_out[ESN_obj.W_out>0.5])))
        # print("Max of W_out is "+str(np.max(ESN_obj.W_out)))
        # Plotting.plot_activations(state_target, ESN_obj.x)
        params = [scaling_W_in,
                  scaling_W,
                  scaling_alpha,
                  beta,
                  scaling_W_fb]
        if save_or_display is not None:
            if system_name == "NaKL":
                Plotting.plot_NaKL(state, state_target, train_start_timestep, train_end_timestep, system_name, dims,
                                     dimension_directory, timesteps_for_prediction, setup_number, perform_grid_search,
                                     params, save_or_display, data_storage_number)
            else:
                Plotting.plot_orbits(state, state_target, train_start_timestep, train_end_timestep, system_name, dims,
                                     dimension_directory, timesteps_for_prediction, setup_number, perform_grid_search,
                                     params, save_or_display, data_storage_number)
        # if "save" in save_or_display.lower():
        #     np.savetxt("states/" + system_name + "/" +
        #            "prediction/"+dimension_directory+
        #            "_orbit_params_(" +
        #            str(round(params[0], 4)) + "," +
        #            str(round(params[1], 4)) + "," +
        #            str(round(params[2], 4)) + "," +
        #            "{:.2e}".format(params[3])+ ","+
        #            str(round(params[4], 4)) + ").txt",
        #            state[:,:train_end_timestep + timesteps_for_prediction])
