import numpy as np
# TODO: Optimize this later

def Set_Grid(state_target, perform_grid_search, setup_number):
    # For parameter search by grid search:
    if setup_number == -1:
        # "Very quick test grid search setup. Not meant as a serious search"
        extra_W_in_scale_factor_grid = np.linspace(0.25, 1.0, 1) # input scalings grid, makes no difference for L96?
        scaling_W_grid = np.linspace(0.3, 1.3, 1) # direct multiplier for spectral radius grid after normalization occurs
        alpha_grid = np.linspace(0.1, 1.0, 1) # uniform leaking rate grid
        # Secondary parameters to grid search
        beta_grid = np.logspace(-4, 1, 2)
        extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid

    if setup_number == 1:
        # "Broad Grid Search for General System Investigation"
        extra_W_in_scale_factor_grid = np.linspace(0.25, 1.0, 5) # input scalings grid, makes no difference for L96?
        scaling_W_grid = np.linspace(0.3, 1.3, 10) # direct multiplier for spectral radius grid after normalization occurs
        alpha_grid = np.linspace(0.1, 1.0, 9) # uniform leaking rate grid
        # Secondary parameters to grid search
        beta_grid = np.logspace(-4, 1, 5)
        extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid

    if setup_number == 2:
        # "Colpitts-Specific Grid Search"
        # Estimate 7.2 hour completion
        extra_W_in_scale_factor_grid = np.linspace(0.01, 1.0, 6) # input scalings grid, makes no difference for L96?
        scaling_W_grid = np.linspace(0.1, 1.0, 16) # direct multiplier for spectral radius grid after normalization occurs
        alpha_grid = np.linspace(0.1, 1.0, 9) # uniform leaking rate grid
        # Secondary parameters to grid search
        beta_grid = np.logspace(-2, 1.5, 6)
        extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid


    # if setup_number == 2:
    #     # "L63 Low Spectral Radius Specific Grid Search"
    #     extra_W_in_scale_factor_grid = np.linspace(0.25, 1.0, 5) # input scalings grid, makes no difference for L96?
    #     scaling_W_grid = np.linspace(0.3, 0.95, 7) # direct multiplier for spectral radius grid after normalization occurs
    #     alpha_grid = np.linspace(0.6, 0.95, 9) # uniform leaking rate grid
    #     # Secondary parameters to grid search
    #     beta_grid = np.logspace(-5, -3, 6)
    #     extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid
    #
    # if setup_number == 3:
    #     # "L63 High Spectral Radius Specific Grid Search"
    #     extra_W_in_scale_factor_grid = np.linspace(1.0, 1.7, 5) # input scalings grid, makes no difference for L96?
    #     scaling_W_grid = np.linspace(1.0, 1.5, 5) # direct multiplier for spectral radius grid after normalization occurs
    #     alpha_grid = np.linspace(0.6, 0.95, 9) # uniform leaking rate grid
    #     # Secondary parameters to grid search
    #     beta_grid = np.logspace(-5, -3, 6)
    #     extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid
    #
    # if setup_number == 4:
    #     # "Colpitts-Specific Grid Search"
    #     extra_W_in_scale_factor_grid = np.linspace(0.1, 0.7, 5) # input scalings grid, makes no difference for L96?
    #     scaling_W_grid = np.linspace(0.5, 1.3, 5) # direct multiplier for spectral radius grid after normalization occurs
    #     alpha_grid = np.linspace(0.5, 0.9, 9) # uniform leaking rate grid
    #     # Secondary parameters to grid search
    #     beta_grid = np.logspace(-3, -1, 6)
    #     extra_W_fb_scale_factor_grid = np.array(range(1,2))/1.0 # input scalings grid
    #
    # if setup_number == 5:
    #     # "Colpitts-Specific Grid Search To Test Ten Different Alpha Variations, Holding All Other Parameters Fixed"
    #     extra_W_in_scale_factor_grid = np.array([0.1]) # input scalings grid, makes no difference for L96?
    #     scaling_W_grid = np.array([0.9]) # direct multiplier for spectral radius grid after normalization occurs
    #     alpha_grid = np.multiply(np.divide(np.arange(0.0,10.0,1),50), 0.8) # uniform leaking rate grid
    #     # Secondary parameters to grid search
    #     beta_grid = np.array([1.0])
    #     extra_W_fb_scale_factor_grid = np.array([1.0]) # input scalings grid

    if perform_grid_search:
        list_of_W_in_scale_factor = extra_W_in_scale_factor_grid
        list_of_scaling_W = scaling_W_grid
        list_of_scaling_alpha = alpha_grid
        list_of_beta_to_test = beta_grid
        list_of_scaling_W_fb = extra_W_fb_scale_factor_grid
        list_of_scaling_W_in = list_of_W_in_scale_factor * np.max(np.abs(state_target))
    else:
        # L63
        list_of_W_in_scale_factor = [0.25]
        list_of_scaling_W = [0.8]
        list_of_scaling_alpha = [0.9062]
        list_of_beta_to_test = [0.0001]
        list_of_scaling_W_fb = [1.0]
        list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))
        # L63
        # list_of_W_in_scale_factor = [0.25]
        # list_of_scaling_W = [0.8]
        # list_of_scaling_alpha = [0.9062]
        # list_of_beta_to_test = [0.0001]
        # list_of_scaling_W_fb = [1.0]
        # list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))
        # # L96 9D and 10D
        # list_of_W_in_scale_factor = [1.0]
        # list_of_scaling_W = [0.8556]
        # list_of_scaling_alpha = [0.55]
        # list_of_beta_to_test = [0.0316]
        # list_of_scaling_W_fb = [1.0]
        # list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))
        # # L96 8D
        # list_of_W_in_scale_factor = [0.8125]
        # list_of_scaling_W = [1.0778]
        # list_of_scaling_alpha = [0.4375]
        # list_of_beta_to_test = [0.0316]
        # list_of_scaling_W_fb = [1.0]
        # list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))
        # L96 7D
        # list_of_W_in_scale_factor = [1.0]
        # list_of_scaling_W = [0.7444]
        # list_of_scaling_alpha = [0.4375]
        # list_of_beta_to_test = [0.0316]
        # list_of_scaling_W_fb = [1.0]
        # list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))
        # Colpitts
        # list_of_W_in_scale_factor = [0.1]
        # list_of_scaling_W = [0.8]
        # list_of_scaling_alpha = [0.9]
        # list_of_beta_to_test = [0.1]
        # list_of_scaling_W_fb = [1.0]
        # list_of_scaling_W_in = np.array(list_of_W_in_scale_factor) * np.max(np.abs(state_target))

    return list_of_scaling_W, list_of_scaling_alpha, list_of_beta_to_test, \
           list_of_scaling_W_fb, list_of_scaling_W_in
