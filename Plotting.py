import matplotlib.pyplot as plt
import numpy as np
def plot_3D_orbits(Y, Y_target, train_end_timestep):
    # Plot Y and Y_target for comparison:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Y_target[0, :train_end_timestep].transpose(), Y_target[1, :train_end_timestep].transpose(), Y_target[2, :train_end_timestep].transpose())
    ax.plot(Y[0, train_end_timestep:].transpose(), Y[1, train_end_timestep:].transpose(), Y[2, train_end_timestep:].transpose())
    plt.show()

def plot_activations(Y, x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_x = []
    y_x = []
    z_x = []
    counter1 = 0
    counter2 = 0
    for ind_i in range(np.shape(x)[0]):
        for ind_j in range(np.shape(x)[1]):
            if counter1 % 100 == 0 and counter2 % 10 == 0:
                z_x.append(x[ind_i, ind_j])
                x_x.append(ind_i)
                y_x.append(ind_j)
            counter2 += 1
        counter1 += 1
    ax.scatter(np.array(x_x), np.array(y_x), np.array(z_x))
    # Make second plot
    plt.figure()
    plt.plot(Y.transpose())
    plt.show()