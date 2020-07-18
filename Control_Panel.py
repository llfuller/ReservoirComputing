import Lorenz63
import Lorenz96
import Colpitts
import Grid_Search_Settings
import ESN_Process
import ESN
import numpy as np
import time

"""
This script is the only script to run. It uses the other scripts in the directory to run sub-tasks.
Parameters are specified at the top.
This is used to create graphs of outputs for single ESNs and grid search.
"""

np.random.seed(2020)
#=======================================================================================================================
# Run Parameters
#=======================================================================================================================
system_name = "L96"
run_system = False # Generate new data from chosen system
N_x = 1000 # Number of nodes in reservoir."should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
perform_grid_search = False
sparsity_tuples = np.array([[2.0/N_x,0.95],
                            [20.0/N_x,0.05]
                            ])
# First value: sparisty (numerator is average number of connections FROM one node TO other nodes),
# second value: proportion of network with that sparsity
sparsity = 15.0 / N_x # Only applies to GPU so far TODO: What if symmetric?
train_start_timestep = 2000
train_end_timestep = 10000 # Timestep at which training ends.
timesteps_for_prediction = 1000 # if this is too big, then MSE becomes almost meaningless. Too small and you can't tell
# what the overall prediction behavior is.
save_or_display = "3d display"  #save 3d or 3d plots of orbits after prediction or display them. Set to None for neither.
# use 3d or 2d prefix for either type of graph.
print_timings_boolean = False
# Since all data is normalized, the characteristic length is 1. I'll set the allowed deviation length to 0.05 of this.
save_name = "ESN_1"
dev_length_multiplier = 2.0


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
state = np.empty((N_y, train_end_timestep+timesteps_for_prediction)) # Input to reservoir. Before train_end_timestep,
# state is identical to state_target. After that index, it will differ as this is a prediction of the state by the
# reservoir.
# Lorenz96.plot_L96()
#=======================================================================================================================
# Grid Search Info
#=======================================================================================================================
i,j,k,l,m = 0,0,0,0,0 # Indices to use in each grid if not grid searching
# TODO: Revise list_of_scaling_W_in and list_of_W_in_scale_factor usage after this point (should be good though) compare
list_of_scaling_W, list_of_scaling_alpha, list_of_beta_to_test, list_of_scaling_W_fb, \
list_of_scaling_W_in = Grid_Search_Settings.Set_Grid(state_target, perform_grid_search)
# These values are used if no looping over that list:
scaling_W = list_of_scaling_W[0]
scaling_W_in = list_of_scaling_W_in[0]
scaling_W_fb = list_of_scaling_W_fb[0]
scaling_alpha = list_of_scaling_alpha[0]

# Keeping track of mse:
mse_array = 100000000*np.ones((np.shape(list_of_scaling_W_in)[0],
                      np.shape(list_of_scaling_W)[0],
                      np.shape(list_of_scaling_alpha)[0],
                      np.shape(list_of_beta_to_test)[0]))

#=======================================================================================================================
# Construct, Train, and Predict
#=======================================================================================================================
x_initial = np.random.rand(N_x)

# Construct ESN (parameters will be reset in the loop later on)
print("Now building ESN at time " + str(time.time() - start_time))
placeholder_array = np.ones(1) # will be replaced later
if N_x <= 6000:
    ESN_Build_Method = ESN.ESN_CPU
    print("Using CPU")
else:
    ESN_Build_Method = ESN.ESN_GPU
    print("Using GPU")

ESN_1 = ESN_Build_Method(N_x, N_u, N_y, sparsity_tuples,
                         x_initial, scaling_alpha * np.ones(N_x), scaling_W,
                         scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction)
ESN_list = [ESN_1]
for i in range(10):
    ESN_list.append(ESN_Build_Method(N_x, N_u, N_y, sparsity_tuples,
                         x_initial, float(i)/50*scaling_alpha * np.ones(N_x), scaling_W,
                         scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction))
Group_1 = ESN.Reservoir_Group(ESN_list)

print("Done building/loading ESN at time " + str(time.time() - start_time))
# Training and Prediction
for i, scaling_W_in in enumerate(list_of_scaling_W_in):
    for j, scaling_W_fb in enumerate(list_of_scaling_W_fb):
        for k, scaling_alpha in enumerate(list_of_scaling_alpha):
            # The beta loop is located inside ESN_Process because that is more efficient
            print("------------------\n")
            ESN_Process.build_and_train_and_predict(Group_1,
                                                    start_time, train_start_timestep, train_end_timestep,
                                                    mse_array, list_of_beta_to_test, N_u, N_y, N_x, x_initial,
                                                    state_target,
                                                    scaling_W_fb, timesteps_for_prediction, scaling_W_in,
                                                    system_name, print_timings_boolean, scaling_alpha,
                                                    scaling_W, save_or_display,
                                                    state, save_name, param_array=[i,j,k,l,m])
print("Minimum MSE of " +str(mse_array.min()))
indices_of_min = np.unravel_index(mse_array.argmin(), mse_array.shape)
print("Min MSE at parameter indices: "+str(indices_of_min))
print("Min MSE at parameters: "+"("+str(list_of_scaling_W_in[indices_of_min[0]])+","+
      str(list_of_scaling_W[indices_of_min[1]])+","+
      str(list_of_scaling_alpha[indices_of_min[2]])+","+
      str(list_of_beta_to_test[indices_of_min[3]])+","+
      "scaling_W_fb="+str(scaling_W_fb)+")")
np.savez("mse_array_"+str(N_x)+".npz",mse_array=mse_array)
print("Done at time: "+str(time.time()-start_time))