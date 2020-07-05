import Lorenz63
import ESN
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
import scipy as sp

start_time = time.time()
np.random.seed(2020)
testSystem = "L63"
train_start_timestep = 3000
num_timesteps_train = 100000 # arbitrarily chosen number
# For parameter search by grid search:

extra_W_in_scale_factor_grid = np.float32(range(1,1000))/5.0 # input scalings grid
scaling_W_grid = np.float32(range(1,6))/5.0 # multiplier for spectral radius grid
alpha_grid = np.float32(range(11))/10.0 # uniform leaking rate grid

# Secondary parameters to optimize
beta_grid = np.logspace(-4, 6, 10)

# Keeping track of mse:
mse_array = 100000000*np.ones((np.shape(extra_W_in_scale_factor_grid)[0],
                      np.shape(scaling_W_grid)[0],
                      np.shape(alpha_grid)[0],
                      np.shape(beta_grid)[0]))

# scaling_W is for tuning procedure after normalization of W
# scaling_W = 1 #TBD in tuning
beta = 20 # regularization coefficient (Lukosevicius PracticalESN Eqtn 9)

if testSystem is "L63":
    t_final = 1000.0
    dt = 0.001
    Lorenz63.run_L63(t_final, dt)
    # N_x "should be at least equal to the estimate of independent real values
    # the reservoir has to remember from the input to solve its task"
    # -Lukosevicius in PracticalESN
    N_u = Lorenz63.setup_ESN_params_L63()[0]
    N_y = Lorenz63.setup_ESN_params_L63()[1]
    Y_target = (np.loadtxt('L63_States.txt')).transpose()
    # print("Shape of np.shape(Y_target)")
    # print(np.shape(Y_target))
    num_timesteps_data = np.shape(Y_target)[1]
Y = np.empty((N_y, num_timesteps_train))
N_x = 3 * 120
x_initial = np.random.rand(N_x)

def plot_3D_orbits(num_timesteps, Y, Y_target):
    # Plot Y and Y_target for comparison:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Y_target[0, :num_timesteps_train].transpose(), Y_target[1, :num_timesteps_train].transpose(), Y_target[2, :num_timesteps_train].transpose())
    ax.plot(Y[0, :].transpose(), Y[1, :].transpose(), Y[2, :].transpose())
    # ax = fig.gca()
    # ax.plot(Y_target[1,:num_timesteps_train])
    # ax.plot(Y[2, :num_timesteps_train])
    # ax.plot(x)
    # plt.plot
    plt.show()

def main_method(start_time,train_start_timestep,num_timesteps_train,mse_array,beta,
                N_u,N_y,N_x,x_initial,Y_target,num_timesteps_data):
    print("Testing with " + str((extra_W_in_scale_factor, scaling_W, scaling_alpha)))
    scaling_W_in = extra_W_in_scale_factor * np.max(Y_target)  # normalization factor for inputs
    alpha_input = scaling_alpha * np.ones(N_x)
    sparsity = 10.0 / N_x
    after_system_sim_time = time.time() - start_time
    print(after_system_sim_time)
    print("after_system_sim_time")
    # x_initial is random since I don't know any better
    x = np.zeros((num_timesteps_train, N_x))
    x[0] = x_initial

    # Construct ESN architecture
    ESN_1 = ESN.ESN(N_x, N_u, N_y, sparsity,
                    x_initial, alpha_input, scaling_W,
                    scaling_W_in)
    print("W_in: " + str(ESN_1.W_in))

    after_ESN_construction_time = time.time() - start_time
    print(after_ESN_construction_time)
    print("after_ESN_construction_time")
    print("W: " + str(ESN_1.W))

    # Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
    ESN_1.x = x[0]
    for n in range(1, num_timesteps_train):
        ESN_1.update_reservoir(Y_target[:, n], x[n - 1])
        # print(ESN_1.x)
        x[n] = ESN_1.x

    after_ESN_feed_time = time.time() - start_time
    print(after_ESN_feed_time)
    print("after_ESN_feed_time")

    # Compute W_out (readout coefficients)
    ESN_1.calculate_W_out(Y_target[:, train_start_timestep:num_timesteps_train],
                          x[train_start_timestep:], N_x, beta, train_start_timestep,
                          num_timesteps_train)

    after_W_out_train_time = time.time() - start_time
    print(after_W_out_train_time)
    print("after_W_out_train_time")

    # Predict Y at each training timestep:
    Y[:, 0] = Y_target[:, 0]
    ESN_1.x = x[0]
    x_predict = np.empty(np.shape(x))
    x_predict[0] = x[0]
    for n in range(1, num_timesteps_train):
        Y[:, n] = ESN_1.output_Y(Y[:, n - 1], x_predict[n - 1])
        ESN_1.update_reservoir(Y[:, n], x_predict[n - 1])
        x_predict[n] = ESN_1.x

    after_Y_predict_train_time = time.time() - start_time
    print(after_Y_predict_train_time)

    print(np.shape(x))
    np.savez('ESN_1', ESN_1=ESN_1)

    # print("Ratio squared error is: " + str(calculate_ratio_squared_error(Y,Y_target[:,:num_timesteps_train])))
    print("mean_squared_error is: " + str(mean_squared_error(Y.transpose(),
                                                             Y_target[:, :num_timesteps_train].transpose())))
    mse_array[i, j, k, l] = mean_squared_error(
        Y.transpose(),
        Y_target[:, :num_timesteps_train].transpose())
    print("W_out: " + str(ESN_1.W_out))
    print("Max of W_out is "+str(np.max(ESN_1.W_out)))
    plt.figure()
    plt.plot(ESN_1.W_out)
i,j,k, l = 9,4,9,6

extra_W_in_scale_factor = extra_W_in_scale_factor_grid[i]
scaling_W = scaling_W_grid[j]
scaling_alpha = alpha_grid[k]
beta = beta_grid[l]
main_method(start_time,train_start_timestep,num_timesteps_train,mse_array,beta,
                N_u,N_y,N_x,x_initial,Y_target,num_timesteps_data)
plot_3D_orbits(num_timesteps_train, Y, Y_target)
# for i, extra_W_in_scale_factor in enumerate(extra_W_in_scale_factor_grid):
#     for j, scaling_W in enumerate(scaling_W_grid):
#         for k, scaling_alpha in enumerate(alpha_grid):
#             main_method(start_time,train_start_timestep,num_timesteps_train,mse_array,beta,
#                 N_u,N_y,N_x,x_initial,Y_target,num_timesteps_data)
print("Minimum MSE of " +str(mse_array.min()))
print("Min MSE at parameters: "+str(np.unravel_index(mse_array.argmin(), mse_array.shape)))
# print("mse_array: "+str(mse_array))
