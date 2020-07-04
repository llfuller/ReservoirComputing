import Lorenz63
import ESN
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse


np.random.seed(2020)
testSystem = "L63"
num_timesteps = 200 # arbitrarily chosen number
# scaling_W is for tuning procedure after normalization of W
scaling_W = 1 #TBD in tuning
sparsity = 0.1

# N_x "should be at least equal to the estimate of independent real values
# the reservoir has to remember from the input to solve its task"
# -Lukosevicius in PracticalESN
if testSystem is "L63":
    N_x = 3 * 10
    N_u = Lorenz63.setup_ESN_params_L63()[0]
    N_y = Lorenz63.setup_ESN_params_L63()[1]
    Y_target = np.loadtxt('L63_States.txt')
    scaling_W_in = np.max(Y_target) # normalization factor for inputs
    alpha_input = 0.001 * np.ones(N_x)
    num_timesteps_data = np.shape(Y_target)[0]

# x_initial is random since I don't know any better
x_initial = np.random.rand(N_x)
x = -1234*np.ones((num_timesteps, N_x))

# Construct ESN architecture
ESN_1 = ESN.ESN(N_x, N_u, N_y, sparsity,
                x_initial, alpha_input, scaling_W,
                scaling_W_in)
ESN_1.build_W(N_x, sparsity, scaling_W)
ESN_1.build_W_in(N_x, N_u, scaling_W_in)

# Run ESN for however many timesteps necessary to get enough elements in Y
x[0] = x_initial
for n in range(1,num_timesteps):
    ESN_1.update_reservoir(Y_target[n-1])
    x[n] = ESN_1.x

ESN_1.calculate_W_out(Y_target[n - 1], ESN_1.x)

# Create empty ESN output matrix Y with dim(Y)=dim(Y_target)
Y = np.empty_like(Y_target)

# Predict next few timesteps:
ESN_1.x = x[num_timesteps-1]
for n in range(num_timesteps, num_timesteps_data):
    ESN_1.update_reservoir(Y_target[n-1])
    Y[n] = ESN_1.output_Y(Y_target[n-1])

# Plot Y and Y_target for comparison:
fig = plt.figure()
ax = fig.gca()
ax.plot(Y_target)
plt.plot()
plt.show()
