import numpy as np
# TODO: Optimize this later

def Set_Grid(state_target, perform_grid_search):
    # For parameter search by grid search:
    extra_W_in_scale_factor_grid = np.float32(range(1,5))/2.0 # input scalings grid, makes no difference for L96?
    scaling_W_grid = np.float32(range(1,4))/3.0 # direct multiplier for spectral radius grid after normalization occurs
    alpha_grid = np.float32(range(5,7))/10.0 # uniform leaking rate grid
    # Secondary parameters to grid search
    beta_grid = np.logspace(-2, -0, 2)
    extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid

    if perform_grid_search:
        list_of_W_in_scale_factor = extra_W_in_scale_factor_grid
        list_of_scaling_W = scaling_W_grid
        list_of_scaling_alpha = alpha_grid
        list_of_beta_to_test = beta_grid
        list_of_scaling_W_fb = extra_W_fb_scale_factor_grid
        list_of_scaling_W_in = list_of_W_in_scale_factor * np.max(state_target)
    else:
        list_of_W_in_scale_factor = [0.5]
        list_of_scaling_W = [0.175]
        list_of_scaling_alpha = [1.0]
        list_of_beta_to_test = [0.001]
        list_of_scaling_W_fb = [1.0]
        list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(state_target)

    return list_of_scaling_W, list_of_scaling_alpha, list_of_beta_to_test, \
           list_of_scaling_W_fb, list_of_scaling_W_in
