import numpy as np
import Grid_Search_Settings
import time
import Lorenz96
import Lorenz63
import Colpitts
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
Discover at which time predictions error crosses a threshold and log it.
"""

system_name = "L63"
perform_grid_search = True
run_system = False
setup_number = 1
train_start_timestep = 0
train_end_timestep = 5000 # Timestep at which training ends.
timesteps_for_prediction = 400 # if this is too big, then calculations take too long. Medium range: MSE is meaningless. Short range: good measure
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
for i, scaling_W_in in enumerate(list_of_scaling_W_in):
    for j, scaling_W in enumerate(list_of_scaling_W):
        print("scaling_W = "+str(scaling_W))
        for k, scaling_alpha in enumerate(list_of_scaling_alpha):
            for l, beta in enumerate(list_of_beta_to_test):
                for m, scaling_W_fb in enumerate(list_of_scaling_W_fb):
                    index1 = j
                    index2 = m + l*length_of_scaling_W_fb_list + k*length_of_beta_to_test_list + \
                             i*length_of_scaling_alpha_list
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
                    print("Shape of state array: " + str(state))
                    # For each timestep:
                    for n in range(train_end_timestep+1, train_end_timestep+timesteps_for_prediction):
                        current_errors[index1, index2, n] = mean_squared_error(
                            state_target_transposed[train_end_timestep:n],
                            state.transpose()[train_end_timestep:n])
                        # print("current error: " +str(current_errors[index1, index2, n]))
                        cumulative_errors[index1, index2, n] += np.sum(current_errors[index1, index2, :n])
                    plt.figure()
                    pltx = np.arange(0, train_end_timestep+timesteps_for_prediction)
                    plty = np.sum(cumulative_errors[index1], axis=0)
                    print("pltx: \n"+str(pltx))
                    print("plty: \n"+str(plty))
                    plt.plot(pltx, plty)
                    plt.xlim(train_end_timestep, train_end_timestep+timesteps_for_prediction)
                    plt.savefig("cumulative_error_plots/"+
                                       "_orbit_params_(" +
                                       str(round(params[0], 4)) + "," +
                                       str(round(params[1], 4)) + "," +
                                       str(round(params[2], 4)) + "," +
                                       "{:.2e}".format(params[3])+ ","+
                                       str(round(params[4], 4)) + ").png")
                    # plt.show()