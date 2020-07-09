import Lorenz63
import Lorenz96
import Colpitts
import ESN_Process
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
system_name = "L63"
run_system = False # Generate new data from chosen system
N_x = 360 # Number of nodes in reservoir."should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
perform_grid_search = True
sparsity = 5.0 / N_x # TODO: What if symmetric?
train_start_timestep = 2000
train_end_timestep = 30000 # Timestep at which training ends.
timesteps_for_prediction = 30000 # if this is too big, then MSE becomes almost meaningless. Too small and you can't tell
# what the overall prediction behavior is.
save_or_display = '2d display' #save 3d or 3d plots of orbits after prediction or display them. Set to None for neither.
# use 3d or 2d prefix for either type of graph.
print_timings_boolean = False

start_time = time.time()

#=======================================================================================================================
# Data Input
#=======================================================================================================================
if system_name is "L63":
    if run_system:
        Lorenz63.run_L63(t_final = 1000.0,
                         dt = 0.001)
    N_u, N_y = 3, 3
    state_target = (np.loadtxt('L63_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

if system_name is "L96":
    if run_system:
        Lorenz96.run_L96(t_final = 1000.0,
                         dt = 0.001)
    N_u, N_y = 36, 36
    state_target = (np.loadtxt('L96_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

if system_name is "Colpitts":
    if run_system:
        Colpitts.run_Colpitts(t_final = 10000.0,
                              dt = 0.001)
    N_u, N_y = 3, 3
    state_target = (np.loadtxt('Colpitts_States.txt')).transpose()
    num_timesteps_data = np.shape(state_target)[1]

state_target = np.divide(state_target,np.max(np.abs(state_target))) # Actual input to reservoir.
state = np.empty((N_y, train_end_timestep+timesteps_for_prediction)) # Input to reservoir. Before train_end_timestep,
# state is identical to state_target. After that index, it will differ as this is a prediction of the state by the
# reservoir.
# Since all data is normalized, the characteristic length is 1. I'll set the allowed deviation length to 0.05 of this.
dev_length_multiplier = 0.4

#=======================================================================================================================
# Grid Search Info
#=======================================================================================================================
i,j,k,l,m = 4,2,3,0,0 # Indices to use in each grid if not grid searching
# For parameter search by grid search:
extra_W_in_scale_factor_grid = np.float32(range(1,9))/10.0 # input scalings grid
scaling_W_grid = np.float32(range(5,8))/40.0 # direct multiplier for spectral radius grid after normalization occurs
alpha_grid = np.float32(range(1,6))/100.0 # uniform leaking rate grid
# Secondary parameters to grid search
beta_grid = np.logspace(-9, -3, 10)
extra_W_fb_scale_factor_grid = np.float32(range(1,1000))/5.0 # input scalings grid

# extra_W_in_scale_factor = 0.5 #i
# scaling_W = 0.175 # j, scaling_W is for tuning procedure after normalization of W
# scaling_alpha = 0.04 # k
# beta = 0.00001 # l
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
# Train and Predict
#=======================================================================================================================
x_initial = np.random.rand(N_x)
if perform_grid_search:
    list_of_beta_to_test = beta_grid
    for i, extra_W_in_scale_factor in enumerate(extra_W_in_scale_factor_grid):
        for j, scaling_W in enumerate(scaling_W_grid):
            for k, scaling_alpha in enumerate(alpha_grid):
                # The beta loop is located inside ESN_Process because that is more efficient
                print("------------------\n"
                "Testing for "+str((extra_W_in_scale_factor,scaling_W,scaling_alpha,beta,scaling_W_fb)))
                print("Has Indices: "+str((i,j,k,l,0)))
                ESN_Process.build_and_train_and_predict(start_time, train_start_timestep, train_end_timestep,
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
    ESN_Process.build_and_train_and_predict(start_time, train_start_timestep, train_end_timestep,
                                                        mse_array, list_of_beta_to_test, N_u, N_y, N_x, x_initial,
                                                        state_target,
                                                        scaling_W_fb, timesteps_for_prediction, scaling_W_in,
                                                        system_name, print_timings_boolean, scaling_alpha,
                                                        scaling_W, extra_W_in_scale_factor, save_or_display,
                                                        state, sparsity, dev_length_multiplier, perform_grid_search,
                                                        param_array=[i,j,k,l,m])