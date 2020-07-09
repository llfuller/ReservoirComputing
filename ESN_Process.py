import time
import numpy as np
import Plotting
import ESN
from sklearn.metrics import mean_squared_error


def print_timing(print_timings_boolean, start_time, variable1_str):
    if print_timings_boolean==True:
        print(variable1_str+str(time.time()-start_time)+"\n-----------------------")


def build_and_train_and_predict(start_time,train_start_timestep,train_end_timestep,mse_array,list_of_beta_to_test,
                                N_u,N_y,N_x,x_initial,state_target, scaling_W_fb, timesteps_for_prediction,
                                scaling_W_in, system_name, print_timings_boolean, scaling_alpha, scaling_W,
                                extra_W_in_scale_factor, save_or_display, state,sparsity, dev_length_multiplier,
                                perform_grid_search, param_array):
    i,j,k,l,m = param_array
    alpha_input = scaling_alpha * np.ones(N_x)
    print_timing(print_timings_boolean, start_time, "after_system_sim_time")

    # Construct ESN architecture
    ESN_1 = ESN.ESN(N_x, N_u, N_y, sparsity,
                    x_initial, alpha_input, scaling_W,
                    scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction)
    print_timing(print_timings_boolean, start_time, "after_ESN_construction_time")

    # Create "echoes" and record the activations
    # Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
    for n in range(0, train_end_timestep):
        # Using Teacher Forcing method:
        ESN_1.update_reservoir(state_target[:, n], n, state_target[:,n+1])
        # print(ESN_1.W_fb)
    print_timing(print_timings_boolean, start_time, "after_ESN_feed_time")

    worth_predicting = True

    if perform_grid_search:
        dev_length = dev_length_multiplier\
                     *(np.max(state_target[:,train_end_timestep:train_end_timestep+timesteps_for_prediction] -
                                  np.min(state_target[:,train_end_timestep:train_end_timestep+
                                                                           timesteps_for_prediction])))

    for l, beta in enumerate(list_of_beta_to_test):
        # Compute W_out (readout coefficients)
        ESN_1.calculate_W_out(state_target, N_x, beta, train_start_timestep,
                              train_end_timestep)
        print_timing(print_timings_boolean, start_time, "after_W_out_train_time")

        # Predict state at each next timestep, keeping training progress (This isn't training.
        # This is prediction):
        state[:, 0:train_end_timestep+1] = state_target[:, 0:train_end_timestep+1]
        for n in range(train_end_timestep, train_end_timestep + timesteps_for_prediction-1):
            state[:, n+1] = ESN_1.output_Y(state[:, n], n)
            ESN_1.update_reservoir(state[:, n], n, state[:,n+1])
            if perform_grid_search: # This if statement is a trick to avoid unnecessary calculation during grid search
                if np.max(abs(state[:,n+1]-state_target[:,n+1]))>dev_length:
                    print("Not worth predicting for beta="+str(beta))
                    worth_predicting = False
                    break # terminates prediction. Not worth predicting this
        if worth_predicting:
            print_timing(print_timings_boolean, start_time, "after_Y_predict_train_time")

            # np.savez('ESN_1', ESN_1=ESN_1)

            print("mean_squared_error is: " + str(mean_squared_error(
                state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
                state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())))
            mse_array[i, j, k, l] = mean_squared_error(
                state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
                state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())
            print("Number of large W_out:"+str(np.sum(ESN_1.W_out[ESN_1.W_out>0.5])))
            print("Max of W_out is "+str(np.max(ESN_1.W_out)))
            # plt.figure()
            # plt.plot(ESN_1.W_out)
            # Plotting.plot_activations(state_target, ESN_1.x)
            if save_or_display is not None:
                Plotting.plot_orbits(state, state_target, train_start_timestep, train_end_timestep, system_name,
                                        timesteps_for_prediction,
                                        [extra_W_in_scale_factor,
                                         scaling_W,
                                         scaling_alpha,
                                         beta,
                                         scaling_W_fb],
                                        save_or_display)