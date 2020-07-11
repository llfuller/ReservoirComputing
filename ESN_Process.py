import time
import numpy as np
import Plotting
from sklearn.metrics import mean_squared_error


def print_timing(print_timings_boolean, start_time, variable1_str):
    if print_timings_boolean==True:
        print(variable1_str+str(time.time()-start_time)+"\n-----------------------")


def build_and_train_and_predict(ESN_obj, start_time,train_start_timestep,train_end_timestep,mse_array,
                                list_of_beta_to_test, N_u,N_y,N_x,x_initial,state_target, scaling_W_fb,
                                timesteps_for_prediction, scaling_W_in, system_name, print_timings_boolean,
                                scaling_alpha, scaling_W, save_or_display, state, save_name, param_array):
    # First thing: Set parameters of ESN_obj correctly for this run:
    i,j,k,l,m = param_array
    ESN_obj.W_in = ESN_obj.build_W_in(N_x, N_u, scaling_W_in)
    ESN_obj.W_fb = ESN_obj.build_W_fb(N_x, N_u, scaling_W_fb)
    ESN_obj.alpha_matrix = scaling_alpha * np.ones(N_x)

    # Create "echoes" and record the activations
    # Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
    for n in range(0, train_end_timestep):
        # Using Teacher Forcing method:
        ESN_obj.update_reservoir(state_target[:, n], n, state_target[:,n+1])
    print_timing(print_timings_boolean, start_time, "after_ESN_feed_time")

    for l, beta in enumerate(list_of_beta_to_test):
        print("Testing for " + str((scaling_W_in, scaling_W, scaling_alpha, beta, scaling_W_fb)))
        print("Has Indices: " + str((i, j, k, l, 0)))

        # Compute W_out (readout coefficients)
        ESN_obj.calculate_W_out(state_target, N_x, beta, train_start_timestep,
                              train_end_timestep)
        print_timing(print_timings_boolean, start_time, "after_W_out_train_time")

        # Clear activations and state from train_end_timestep onward to make sure they aren't
        # being used improperly during prediction:
        ESN_obj.x[train_end_timestep+1:] = 0
        state[:,train_end_timestep+1:] = 0
        # Make prediction before end of training identical to target state
        state[:, 0:train_end_timestep+1] = state_target[:, 0:train_end_timestep+1]
        # Predict state at each next timestep, keeping training progress:
        for n in range(train_end_timestep, train_end_timestep + timesteps_for_prediction-1):
            state[:, n+1] = ESN_obj.output_Y(state[:, n], n)
            ESN_obj.update_reservoir(state[:, n], n, state[:,n+1])
        print_timing(print_timings_boolean, start_time, "after_Y_predict_train_time")
        np.savez(save_name+'.npz', ESN_obj=ESN_obj)
        print("mean_squared_error is: " + str(mean_squared_error(
            state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
            state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())))
        mse_array[i, j, k, l] = mean_squared_error(
            state_target.transpose()[train_end_timestep:train_end_timestep+timesteps_for_prediction],
            state[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())
        print("Number of large W_out:"+str(np.sum(ESN_obj.W_out[ESN_obj.W_out>0.5])))
        print("Max of W_out is "+str(np.max(ESN_obj.W_out)))
        # Plotting.plot_activations(state_target, ESN_obj.x)
        if save_or_display is not None:
            Plotting.plot_orbits(state, state_target, train_start_timestep, train_end_timestep, system_name,
                                    timesteps_for_prediction,
                                    [scaling_W_in,
                                     scaling_W,
                                     scaling_alpha,
                                     beta,
                                     scaling_W_fb],
                                    save_or_display)