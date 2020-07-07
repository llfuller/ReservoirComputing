import Lorenz63
import Colpitts
import ESN
import Plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.metrics import mean_squared_error
import scipy as sp

np.random.seed(2020)
testSystem = "L63"
perform_grid_search = False
# Indices to use in each grid if not grid searching
i,j,k,l,m = 0,0,0,0,0
print_timings_boolean = False
train_start_timestep = 2000
train_end_timestep = 30000 # Timestep at which training ends.
timesteps_for_prediction = 2000


# N_x "should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
# There are three dependent values in L63 and Colpitts for example, but who knows how many "independent" values.
N_x = 3 * 120


def print_timing(print_timings_boolean, start_time, variable1_str):
    if print_timings_boolean==True:
        print(variable1_str+str(time.time()-start_time)+"\n-----------------------")

def build_and_train_and_predict(start_time,train_start_timestep,train_end_timestep,mse_array,beta,
                N_u,N_y,N_x,x_initial,Y_target, scaling_W_fb, timesteps_for_prediction, scaling_W_in):
    alpha_input = scaling_alpha * np.ones(N_x)
    sparsity = 10.0 / N_x
    print_timing(print_timings_boolean, start_time, "after_system_sim_time")

    # Construct ESN architecture
    ESN_1 = ESN.ESN(N_x, N_u, N_y, sparsity,
                    x_initial, alpha_input, scaling_W,
                    scaling_W_in, scaling_W_fb, train_end_timestep, timesteps_for_prediction)
    print_timing(print_timings_boolean, start_time, "after_ESN_construction_time")

    # Create "echoes" and record the activations
    # Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
    for n in range(1, train_end_timestep):
        # Using Teacher Forcing method:
        ESN_1.update_reservoir(Y_target[:, n], n, Y_target[:,n+1])
        # print(ESN_1.W_fb)
    print_timing(print_timings_boolean, start_time, "after_ESN_feed_time")

    # Compute W_out (readout coefficients)
    ESN_1.calculate_W_out(Y_target[:, train_start_timestep+1:train_end_timestep+1], N_x, beta, train_start_timestep,
                          train_end_timestep)
    print_timing(print_timings_boolean, start_time, "after_W_out_train_time")

    # Predict Y at each next timestep, keeping training progress (this isn't training. this is prediction):
    #
    Y[:, 0:train_end_timestep] = Y_target[:, 0:train_end_timestep]
    for n in range(train_end_timestep, train_end_timestep + timesteps_for_prediction-1):
        Y[:, n+1] = ESN_1.output_Y(Y[:, n], n)
        ESN_1.update_reservoir(Y[:, n], n, 0*Y[:,n+1])
    print_timing(print_timings_boolean, start_time, "after_Y_predict_train_time")

    # np.savez('ESN_1', ESN_1=ESN_1)

    print("mean_squared_error is: " + str(mean_squared_error(
        Y.transpose()[train_end_timestep:],
        Y_target[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())))
    mse_array[i, j, k, l] = mean_squared_error(
        Y.transpose()[train_end_timestep:],
        Y_target[:, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose())
    print("Number of large W_out:"+str(np.sum(ESN_1.W_out[ESN_1.W_out>0.5])))
    print("Max of W_out is "+str(np.max(ESN_1.W_out)))
    # plt.figure()
    # plt.plot(ESN_1.W_out)
    # Plotting.plot_activations(Y, x)


start_time = time.time()
# For parameter search by grid search:
extra_W_in_scale_factor_grid = np.float32(range(1,6))/5.0 # input scalings grid
scaling_W_grid = np.float32(range(1,10))/5.0 # direct multiplier for spectral radius grid after normalization occurs
alpha_grid = np.float32(range(1,21))/20.0 # uniform leaking rate grid
# Secondary parameters to grid search
beta_grid = np.logspace(7, 8, 1)

# Not yet included in grid search or anything like that
#extra_W_fb_scale_factor_grid = np.float32(range(1,1000))/5.0 # input scalings grid


# Keeping track of mse:
mse_array = 100000000*np.ones((np.shape(extra_W_in_scale_factor_grid)[0],
                      np.shape(scaling_W_grid)[0],
                      np.shape(alpha_grid)[0],
                      np.shape(beta_grid)[0]))

# scaling_W is for tuning procedure after normalization of W
# scaling_W = 1 #TBD in tuning
beta = 20 # regularization coefficient (Lukosevicius PracticalESN Eqtn 9)

if testSystem is "L63":
    Lorenz63.run_L63(t_final = 1000.0,
                     dt = 0.001)
    N_u, N_y = 3, 3
    Y_target = (np.loadtxt('L63_States.txt')).transpose()
    num_timesteps_data = np.shape(Y_target)[1]

if testSystem is "Colpitts":
    Colpitts.run_Colpitts(t_final = 10000.0,
                          dt = 0.001)
    N_u, N_y = 3, 3
    Y_target = (np.loadtxt('Colpitts_States.txt')).transpose()
    num_timesteps_data = np.shape(Y_target)[1]

Y = np.empty((N_y, train_end_timestep+timesteps_for_prediction))
x_initial = np.random.rand(N_x)

# These parameters will be reassigned if performing grid search and remain the same otherwise.
extra_W_in_scale_factor = extra_W_in_scale_factor_grid[i]
scaling_W = scaling_W_grid[j]
scaling_alpha = alpha_grid[k]
beta = beta_grid[l]
scaling_W_fb = np.max(Y_target)

scaling_W_in = extra_W_in_scale_factor * np.max(Y_target)  # normalization factor for inputs

if perform_grid_search:
    for i, extra_W_in_scale_factor in enumerate(extra_W_in_scale_factor_grid):
        for j, scaling_W in enumerate(scaling_W_grid):
            for k, scaling_alpha in enumerate(alpha_grid):
                for l, beta in enumerate(beta_grid):
                    print("------------------\n"
                      "Testing for "+str((extra_W_in_scale_factor,scaling_W,scaling_alpha,beta,scaling_W_fb)))
                    print("Testing for "+str((i,j,k,l,0)))
                    build_and_train_and_predict(start_time, train_start_timestep, train_end_timestep, mse_array, beta,
                                            N_u, N_y, N_x, x_initial, Y_target, scaling_W_fb, timesteps_for_prediction, scaling_W_in)
    print("Minimum MSE of " +str(mse_array.min()))
    print("Min MSE at parameters: "+str(np.unravel_index(mse_array.argmin(), mse_array.shape)))
    np.savez("mse_array_"+str(N_x)+".npz",mse_array=mse_array)
else: # Using selected i,j,k,l at start of script
    build_and_train_and_predict(start_time, train_start_timestep, train_end_timestep, mse_array, beta,
                N_u, N_y, N_x, x_initial, Y_target, scaling_W_fb, timesteps_for_prediction, scaling_W_in)
    Plotting.plot_3D_orbits(Y, Y_target, train_end_timestep)
