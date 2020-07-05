import Lorenz63
import ESN
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp

start_time = time.time()
np.random.seed(2020)
testSystem = "L63"
train_start_timestep = 1000
num_timesteps_train = 10000 # arbitrarily chosen number
# scaling_W is for tuning procedure after normalization of W
scaling_W = 1 #TBD in tuning
sparsity = 0.2
beta = 50 # regularization coefficient (Lukosevicius PracticalESN Eqtn 9)
# num_timestep_start = 1000
# N_x "should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
if testSystem is "L63":
    t_final = 1000.0
    dt = 0.01
    Lorenz63.run_L63(t_final, dt)
    N_x = 3 * 2000
    N_u = Lorenz63.setup_ESN_params_L63()[0]
    N_y = Lorenz63.setup_ESN_params_L63()[1]
    Y_target = (np.loadtxt('L63_States.txt')).transpose()
    scaling_W_in = np.max(Y_target) # normalization factor for inputs
    alpha_input = 0.5 * np.ones(N_x)
    # print("Shape of np.shape(Y_target)")
    # print(np.shape(Y_target))
    num_timesteps_data = np.shape(Y_target)[1]

after_system_sim_time = time.time()-start_time
print(after_system_sim_time)
print("after_system_sim_time")
# x_initial is random since I don't know any better
x_initial = np.random.rand(N_x)
x = np.zeros((num_timesteps_train, N_x))
x[0] = x_initial

# Construct ESN architecture
ESN_1 = ESN.ESN(N_x, N_u, N_y, sparsity,
                x_initial, alpha_input, scaling_W,
                scaling_W_in)
print("W_in: " +str(ESN_1.W_in))


after_ESN_construction_time = time.time()-start_time
print(after_ESN_construction_time)
print("after_ESN_construction_time")
print("W: "+str(ESN_1.W))

# Run ESN for however many timesteps necessary to get enough activation elements x of reservoir
ESN_1.x = x[0]
for n in range(1,num_timesteps_train):
    ESN_1.update_reservoir(Y_target[:,n], x[n-1])
    # print(ESN_1.x)
    x[n] = ESN_1.x

after_ESN_feed_time = time.time()-start_time
print(after_ESN_feed_time)
print("after_ESN_feed_time")

# Compute W_out (readout coefficients)
ESN_1.calculate_W_out(Y_target[:,train_start_timestep:num_timesteps_train],
                      x[train_start_timestep:], N_x, beta, train_start_timestep,
                      num_timesteps_train)

after_W_out_train_time = time.time()-start_time
print(after_W_out_train_time)
print("after_W_out_train_time")



# Predict Y at each training timestep:
Y = np.empty((N_y, num_timesteps_train))
Y[:,0] = Y_target[:,0]
ESN_1.x = x[0]
x_predict = np.empty(np.shape(x))
x_predict[0] = x[0]
for n in range(1,num_timesteps_train):
    Y[:,n] = ESN_1.output_Y(Y[:,n-1], x_predict[n-1])
    ESN_1.update_reservoir(Y[:, n], x_predict[n - 1])
    x_predict[n] = ESN_1.x

after_Y_predict_train_time = time.time()-start_time
print(after_Y_predict_train_time)


print(np.shape(x))
np.savez('ESN_1',ESN_1=ESN_1)

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
