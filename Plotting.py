import matplotlib.pyplot as plt
import numpy as np
def plot_orbits(Y, Y_target, train_start_timestep, train_end_timestep,system_name,timesteps_for_prediction,
                   params,save_or_display="save"):
    # Plot Y and Y_target for comparison:
    fig = plt.figure()
    if '3d' in save_or_display.lower():
        ax = fig.gca(projection='3d')
        ax.plot(Y_target[0, :train_end_timestep].transpose(), Y_target[1, :train_end_timestep].transpose(), Y_target[2, :train_end_timestep].transpose(),
                color = 'k', label = 'Training')
        ax.plot(Y[0, train_end_timestep:].transpose(), Y[1, train_end_timestep:].transpose(), Y[2, train_end_timestep:].transpose(),
                color = 'orange', label = 'Prediction')
        ax.plot(Y_target[0,train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(),
                Y_target[1, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(),
                Y_target[2, train_end_timestep:train_end_timestep+timesteps_for_prediction].transpose(),
                color = 'b', label = 'Target Value')
        plt.title(system_name+" after training timesteps: "+ str(train_start_timestep)+" - "+str(train_end_timestep))

        if "save" in save_or_display.lower():
            plt.savefig("3D_Orbit_Plots/"+
                        system_name + "/" +
                        system_name+
                        "_orbit_params_("+
                        str(round(params[0],4))+ ","+
                        str(round(params[1],4))+ ","+
                        str(round(params[2],4))+ ","+
                        "{:.2e}".format(params[3])+ ","+
                        str(round(params[4],4))+").png")
        plt.legend()
    if '2d' in save_or_display.lower():
        ax = fig.gca()
        ax.plot(Y_target[0, train_end_timestep:train_end_timestep + timesteps_for_prediction].transpose(), color = 'b',
                label = 'Target Value')
        ax.plot(Y[0, train_end_timestep:].transpose(), color = 'orange', label = 'Prediction')
        plt.ylabel("x coord")
        plt.xlabel("time index after training")
        plt.title(
            system_name + " after training timesteps: " + str(train_start_timestep) + " - " + str(train_end_timestep))
        print("Saving for beta = "+str(params[3]))
        print(".....")
        plt.legend()
        if "save" in save_or_display.lower():
            plt.savefig("2D_x_Plots/" +
                        system_name + "/" +
                        system_name +
                        "_orbit_params_(" +
                        str(round(params[0], 4)) + "," +
                        str(round(params[1], 4)) + "," +
                        str(round(params[2], 4)) + "," +
                        "{:.2e}".format(params[3])+ ","+
                        str(round(params[4], 4)) + ").png")
    if "display" in save_or_display:
        plt.show()

def plot_activations(Y, x):
    # Plots reservoir activations strength (z-axis) against time and index.
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