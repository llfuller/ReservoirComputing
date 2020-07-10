import Lorenz63
import Lorenz96
import Colpitts
import ESN_Process
import ESN
import numpy as np
import time

"""
This script is the only script to run. It uses the other scripts in the directory to run sub-tasks.
Parameters are specified at the top.
"""

np.random.seed(2020)
#=======================================================================================================================
# Run Parameters
#=======================================================================================================================
system_name = "L96"
run_system = True # Generate new data from chosen system
N_x = 2000 # Number of nodes in reservoir."should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
perform_grid_search = True
sparsity_tuples = np.array([[2.0/N_x,0.95],
                            [20.0/N_x,0.05]
                            ])
# First value: sparisty (numerator is average number of connections FROM one node TO other nodes),
# second value: proportion of network with that sparsity
sparsity = 15.0 / N_x # Only applies to GPU so far TODO: What if symmetric?
train_start_timestep = 2000
train_end_timestep = 5000 # Timestep at which training ends.
timesteps_for_prediction = 1000 # if this is too big, then MSE becomes almost meaningless. Too small and you can't tell
# what the overall prediction behavior is.
save_or_display = '3d display' #save 3d or 3d plots of orbits after prediction or display them. Set to None for neither.
# use 3d or 2d prefix for either type of graph.
print_timings_boolean = False
# Since all data is normalized, the characteristic length is 1. I'll set the allowed deviation length to 0.05 of this.
dev_length_multiplier = 2.0


start_time = time.time()

#=======================================================================================================================
# Data Input
#=======================================================================================================================
if system_name is "L63":
    if run_system:
        Lorenz63.run_L63(t_final = 20000.0,
                         dt = 0.02)
    N_u, N_y = 3, 3
    state_target = (np.loadtxt('L63_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

if system_name is "L96":
    if run_system:
        Lorenz96.run_L96(t_final = 2000.0,
                         dt = 0.01)
    N_u, N_y = 36, 36
    state_target = (np.loadtxt('L96_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

if system_name is "Colpitts":
    if run_system:
        Colpitts.run_Colpitts(t_final = 10000.0,
                              dt = 0.1)
    N_u, N_y = 3, 3
    state_target = (np.loadtxt('Colpitts_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

state_target = np.divide(state_target,np.max(np.abs(state_target))) # Actual input to reservoir.
state = np.empty((N_y, train_end_timestep+timesteps_for_prediction)) # Input to reservoir. Before train_end_timestep,
# state is identical to state_target. After that index, it will differ as this is a prediction of the state by the
# reservoir.
# Lorenz96.plot_L96()
#=======================================================================================================================
# Grid Search Info
#=======================================================================================================================
i,j,k,l,m = 0,0,0,0,0 # Indices to use in each grid if not grid searching
# For parameter search by grid search:
extra_W_in_scale_factor_grid = np.float32(range(1,5))/5.0 # input scalings grid, makes no difference for L96?
scaling_W_grid = np.float32(range(1,4))/3.0 # direct multiplier for spectral radius grid after normalization occurs
alpha_grid = np.float32(range(5,7))/10.0 # uniform leaking rate grid
# Secondary parameters to grid search
beta_grid = np.logspace(-2, -0, 2)
extra_W_fb_scale_factor_grid = np.float32(range(1,1000))/5.0 # input scalings grid

# extra_W_in_scale_factor = 0.5 #i
# scaling_W = 0.175 # j, scaling_W is for tuning procedure after normalization of W
# scaling_alpha = 0.5 # k
# beta = 0.001 # l
extra_W_in_scale_factor = extra_W_in_scale_factor_grid[i] #i
scaling_W = scaling_W_grid[j] # j, scaling_W is for tuning procedure after normalization of W
scaling_alpha = alpha_grid[k] # k
beta = beta_grid[l] # l
scaling_W_fb = 1.0
scaling_W_in = extra_W_in_scale_factor * np.max(state_target)  # normalization factor for inputs

# Keeping track of mse:
mse_array = 100000000*np.ones((np.shape(extra_W_in_scale_factor_grid)[0],
                      np.shape(scaling_W_grid)[0],
                      np.shape(alpha_grid)[0],
                      np.shape(beta_grid)[0]))

#=======================================================================================================================
# Construct, Train, and Predict
#=======================================================================================================================
x_initial = np.random.rand(N_x)

# Construct ESN (parameters will be reset in the loop later on)
print("Now building ESN at time " + str(time.time() - start_time))
if N_x <= 6000:
    ESN_Build_Method = ESN.ESN_CPU
else:
    ESN_Build_Method = ESN.ESN_GPU #TODO: Update this with sparity_tuples or else it won't work right now
ESN_1 = ESN_Build_Method(N_x, N_u, N_y, sparsity_tuples,
                         x_initial, scaling_alpha * np.ones(N_x), scaling_W,
                         scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction)
print("Done building ESN at time " + str(time.time() - start_time))

# Training and Prediction
if perform_grid_search:
    list_of_beta_to_test = beta_grid
    for i, extra_W_in_scale_factor in enumerate(extra_W_in_scale_factor_grid):
        for j, scaling_W in enumerate(scaling_W_grid):
            for k, scaling_alpha in enumerate(alpha_grid):
                # The beta loop is located inside ESN_Process because that is more efficient
                print("------------------\n")
                ESN_Process.build_and_train_and_predict(ESN_1,
                                                        start_time, train_start_timestep, train_end_timestep,
                                                        mse_array, list_of_beta_to_test, N_u, N_y, N_x, x_initial,
                                                        state_target,
                                                        scaling_W_fb, timesteps_for_prediction, scaling_W_in,
                                                        system_name, print_timings_boolean, scaling_alpha,
                                                        scaling_W, extra_W_in_scale_factor, save_or_display,
                                                        state, sparsity, dev_length_multiplier, perform_grid_search,
                                                        param_array=[i,j,k,l,m])
    print("Minimum MSE of " +str(mse_array.min()))
    indices_of_min = np.unravel_index(mse_array.argmin(), mse_array.shape)
    print("Min MSE at parameter indices: "+str(indices_of_min))
    print("Min MSE at parameters: "+"("+str(extra_W_in_scale_factor_grid[indices_of_min[0]])+","+
          str(scaling_W_grid[indices_of_min[1]])+","+
          str(alpha_grid[indices_of_min[2]])+","+
          str(beta_grid[indices_of_min[3]])+","+
          "scaling_W_fb="+str(scaling_W_fb)+")")
    np.savez("mse_array_"+str(N_x)+".npz",mse_array=mse_array)
else: # Using selected i,j,k,l at start of script
    list_of_beta_to_test = [beta]
    ESN_Process.build_and_train_and_predict(ESN_1, start_time, train_start_timestep, train_end_timestep,
                                            mse_array, list_of_beta_to_test, N_u, N_y, N_x, x_initial,
                                            state_target, scaling_W_fb, timesteps_for_prediction, scaling_W_in,
                                            system_name, print_timings_boolean, scaling_alpha, scaling_W,
                                            extra_W_in_scale_factor, save_or_display, state, sparsity,
                                            dev_length_multiplier, perform_grid_search, param_array=[i,j,k,l,m])
print("Done at time: "+str(time.time()-start_time))