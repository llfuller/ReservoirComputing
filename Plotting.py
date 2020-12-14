import matplotlib.pyplot as plt
import numpy as np
def plot_orbits(Y, Y_target, train_start_timestep, train_end_timestep,system_name, dims, dimension_directory,
                timesteps_for_prediction, setup_number, perform_grid_search, params,save_or_display="save"):
    # Plot Y and Y_target for comparison:
    grid_setup_directory = "" # set like this so it's blank if not a grid search
    if perform_grid_search:
        grid_setup_directory = "grid_setup_" + str(setup_number) +"/"

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
                        grid_setup_directory +
                        dimension_directory +
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
        plt.xlim(0, timesteps_for_prediction)
        plt.title(
            system_name + " after training timesteps: " + str(train_start_timestep) + " - " + str(train_end_timestep))
        print("Saving for beta = "+str(params[3]))
        print(".....")
        plt.legend()
        if "save" in save_or_display.lower():
            plt.savefig("2D_x_Plots/" +
                        system_name + "/" +
                        grid_setup_directory +
                        dimension_directory +
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

def plot_sr_pnz_gen_synch_heatmap(mse_array, pnz_array, sr_array):
    plt.imshow(mse_array, cmap='hot', interpolation='nearest')
    plt.xlabel('pnz')
    plt.ylabel('Spectral Radius')
    plt.show()

def plot_1D_quick(some_data):
    plt.figure()
    plt.plot(some_data)
    plt.show()

def plot_contour(xlist, ylist, z_array, x_label, y_label, title):
    # Assuming inputs are 1D arrays and that a loop over combinations of xlist and ylist elements correspond to zlist
    # elements.
    length_of_x = np.shape(xlist)[0]
    length_of_y = np.shape(ylist)[0]
    print("Length of x in plot_contour:"+str(length_of_x))
    print("Length of y in plot_contour:"+str(length_of_y))
    print("Shape of z_array in plot_contour: "+str(np.shape(z_array)))

    if np.shape(z_array)[0] == length_of_x*length_of_y:
        z_grid = np.array((length_of_x,length_of_y))
        for i, x in enumerate(xlist):
            for j, y in enumerate(ylist):
                z_grid[i,j] = z_array[j+length_of_y*i]
    # if np.shape(z_array) == (length_of_x, length_of_y):
    else:
        z_grid = z_array
    X, Y = np.meshgrid(np.array(xlist), np.array(ylist))
    plt.figure()
    plt.contourf(X, Y, z_grid, cmap="seismic")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar()
    plt.plot()
    # plt.show()
    plt.savefig("Gen_Sync_Plots/"+title)

def plot_scatter(array_1, array_2, title, directory, x_label, y_label):
    plt.figure()
    colors_array = []
    # for i in range(100):
    #     colors_array.append('r')
    # for i in range(4000000-100):
    #     colors_array.append('b')
    plt.scatter(array_1, array_2, s=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    # plt.savefig(directory+title+".png")