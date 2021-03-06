import scipy as sp
import scipy.sparse
import cupy as cp
import os.path
import time

# Cupy resource: https://docs-cupy.chainer.org/en/stable/reference/ndarray.html

"""Build an Echo State Network"""
class ESN_CPU: #
    # connection adjacency matrix W (nonlocal):
    W = sp.zeros((1,1))
    # reservoir nodes' activations:
    x = sp.zeros((1,1))
    # Each alpha acts as basically time between updates:
    alpha_matrix = sp.zeros((1,1))
    # W_in has shape (N_x, N_u + 1)
    W_in = sp.zeros((1,1))
    # W_fb has shape (N_x, N_u)
    W_fb = sp.zeros((1,1))
    # W_out has shape (N_y, 1 + N_x + N_u)
    W_out = sp.zeros((1,1))

    def __init__(self, N_x, N_u, N_y, sparsity_tuples,
                 x_initial, alpha_input, scaling_W,
                 scaling_W_in, scaling_W_fb,
                 train_end_timestep, timesteps_for_prediction):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x)
        # alpha_input has shape (N_x)
        self.W = self.build_W(N_x, sparsity_tuples, scaling_W)#input_W
        self.W_in = self.build_W_in(N_x, N_u, scaling_W_in)#input_W_in
        self.W_fb = self.build_W_fb(N_x, N_u, scaling_W_fb)#input_W_fb
        self.x = sp.zeros((train_end_timestep+ timesteps_for_prediction, N_x))
        self.x[0] = x_initial
        self.alpha_matrix = self.build_alpha_matrix(alpha_input)

    def build_W(self, N_x, sparsity_tuples, scaling_W):
        # N_x integer
        # sparsity between 0 and 1 inclusive
        # scaling_W >= 1
        sparsity_tuples_list = tuple([(a_row[0],a_row[1]) for a_row in sparsity_tuples])
        if os.path.isfile('./W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity_tuples_list)+'.txt'):
            print("Loading W")
            # If file already exists
            W = sp.loadtxt('./W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity_tuples_list)+'.txt')
        else:
            # If file doesn't yet exist
            # No notion of locality here
            # Designate connection or no connection at each element of the (N_x) by (N_x) matrix:
            W_sparsity_list = []
            # Rows used for each sparsity
            rows_used_for_sparsity = (sp.around(sp.multiply(sparsity_tuples[:,1],N_x))).astype(int)
            for i, sparsity_pair in enumerate(sparsity_tuples):
                this_density = sparsity_pair[0]
                W_sparsity_list.append(sp.sparse.random(rows_used_for_sparsity[i], N_x,
                                                        density=this_density).todense())
                    #TODO: Make sure there are no 'holes!' in W
            # Build sparse adjacency matrix W:
            W_unnormalized = sp.multiply(sp.random.choice((-1,1), size=(N_x,N_x)),
                                         sp.vstack(W_sparsity_list))
            # Normalize by largest eigenvalue and additional scaling factor
            # to control decrease of spectral radius.
            spectral_radius = sp.amax(abs(sp.linalg.eigvals(W_unnormalized)))
            print("SPECTRAL RADIUS IS IS "+str(spectral_radius))
            W = sp.multiply( scaling_W, sp.divide(W_unnormalized,spectral_radius) )
            sp.savetxt('W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity_tuples_list)+'.txt',W, fmt='%.4f')
        return W

    def build_W_in(self, N_x, N_u, scaling_W_in):
        W_in = (1.0/scaling_W_in)*sp.random.random((N_x, 1+N_u))
        return W_in

    def build_W_fb(self, N_x, N_u, scaling_W_fb):
        W_fb = (1.0/scaling_W_fb)*sp.random.random((N_x, N_u))
        return W_fb

    def build_alpha_matrix(self, alpha_input):
        return alpha_input

    def update_reservoir(self, u, n, Y):
        # u is input at specific time
        #   u has shape (N_u (3 for L63))
        # See page 16 eqtn 18 of Lukosevicius PracticalESN for feedback info.
        x_n_tilde = sp.tanh(sp.matmul(self.W,self.x[n])
                            + sp.matmul(self.W_in, sp.hstack((sp.array([1]),u)))
                            + sp.matmul(self.W_fb, Y)
                            ) # TODO: Add derivative term?
        self.x[n+1] = sp.multiply((1-self.alpha_matrix), self.x[n]) \
              + sp.multiply(self.alpha_matrix, x_n_tilde)

    def calculate_W_out(self, Y_target, N_x, beta, train_start_timestep, train_end_timestep):
        # see Lukosevicius Practical ESN eqtn 11
        # Using ridge regression
        N_u = sp.shape(Y_target)[0]
        X = sp.vstack((sp.ones((1,train_end_timestep-train_start_timestep)),
                       Y_target[:,train_start_timestep:train_end_timestep],
                       self.x[train_start_timestep:train_end_timestep].transpose()))
        # Ridge Regression
        W_out = sp.matmul(sp.array(Y_target[:, train_start_timestep+1:train_end_timestep+1]),
                          sp.matmul(X.transpose(),
                                    sp.linalg.inv(sp.matmul(X,X.transpose()) + beta*sp.identity(1+N_x+N_u))))
        self.W_out = W_out

    def output_Y(self, u, n):
        one_by_one = sp.array([1])
        # Eqtn 4
        concatinated_matrix = sp.hstack((sp.hstack((one_by_one,
                                                    u)),
                                         self.x[n]))
        return sp.matmul(self.W_out, concatinated_matrix)


class ESN_GPU: #
    """
    Only recommended for large (N_x>6000) reservoirs.
    """
    # connection adjacency matrix W (nonlocal):
    # W is a cupy array
    W = sp.zeros((1,1))
    # reservoir nodes' activations:
    x = sp.zeros((1,1))
    # Each alpha acts as basically time between updates:
    alpha_matrix = sp.zeros((1,1))
    # W_in has shape (N_x, N_u + 1)
    W_in = sp.zeros((1,1))
    # W_fb has shape (N_x, N_u)
    W_fb = sp.zeros((1,1))
    # W_out has shape (N_y, 1 + N_x + N_u)
    # W_out is a cupy array
    W_out = sp.zeros((1,1))

    def __init__(self, N_x, N_u, N_y, sparsity,
                 x_initial, alpha_input, scaling_W,
                 scaling_W_in, scaling_W_fb,
                 train_end_timestep, timesteps_for_prediction):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x)
        # alpha_input has shape (N_x)
        self.W = cp.array(self.build_W(N_x, sparsity, scaling_W))#input_W
        self.W_in = self.build_W_in(N_x, N_u, scaling_W_in)#input_W_in
        self.W_fb = self.build_W_fb(N_x, N_u, scaling_W_fb)#input_W_fb
        self.x = cp.zeros((train_end_timestep+ timesteps_for_prediction, N_x))
        self.x[0] = cp.array(x_initial)
        self.alpha_matrix = self.build_alpha_matrix(alpha_input)

    def build_W(self, N_x, sparsity_tuples, scaling_W):
        # N_x integer
        # sparsity between 0 and 1 inclusive
        # scaling_W >= 1
        sparsity_tuples_list = tuple([(a_row[0], a_row[1]) for a_row in sparsity_tuples])
        if os.path.isfile('./W_(adjacency)/W_' + str(N_x) + '_' + str(N_x) + '_' + str(sparsity_tuples_list) + '.txt'):
            # If file already exists
            W = cp.array(sp.loadtxt('./W_(adjacency)/W_' + str(N_x) + '_' +
                                    str(N_x) + '_' + str(sparsity_tuples_list) + '.txt'))
        else:
            # If file doesn't yet exist
            # No notion of locality here
            # Designate connection or no connection at each element of the (N_x) by (N_x) matrix:
            W_sparsity_list = []
            # Rows used for each sparsity
            rows_used_for_sparsity = (sp.around(sp.multiply(sparsity_tuples[:, 1], N_x))).astype(int)
            for i, sparsity_pair in enumerate(sparsity_tuples):
                this_density = sparsity_pair[0]
                W_sparsity_list.append(sp.sparse.random(rows_used_for_sparsity[i], N_x,
                                                        density=this_density).todense())
                # TODO: Make sure there are no 'holes!' in W
            # Build sparse adjacency matrix W:
            W_unnormalized = cp.multiply(cp.random.choice((-1, 1), size=(N_x, N_x)),
                                         cp.array(sp.vstack(W_sparsity_list)))
            # Normalize by largest eigenvalue and additional scaling factor
            # to control decrease of spectral radius.
            spectral_radius = sp.amax(sp.array(abs(sp.linalg.eigvals(cp.asnumpy(W_unnormalized)))))
            print("SPECTRAL RADIUS IS IS " + str(spectral_radius))
            W = cp.multiply(scaling_W, cp.divide(W_unnormalized, spectral_radius))
            sp.savetxt('W_(adjacency)/W_' + str(N_x) + '_' + str(N_x) + '_' + str(sparsity_tuples_list) + '.txt',
                       cp.asnumpy(W), fmt='%.4f')
        return W

    def build_W_in(self, N_x, N_u, scaling_W_in):
        W_in = (1.0/scaling_W_in)*sp.random.random((N_x, 1+N_u))
        return W_in

    def build_W_fb(self, N_x, N_u, scaling_W_fb):
        W_fb = (1.0/scaling_W_fb)*sp.random.random((N_x, N_u))
        return W_fb

    def build_alpha_matrix(self, alpha_input):
        return cp.array(alpha_input)

    def update_reservoir(self, u, n, Y):
        # u is input at specific time
        #   u has shape (N_u (3 for L63))
        # See page 16 eqtn 18 of Lukosevicius PracticalESN for feedback info.
        x_n_tilde = cp.tanh(cp.matmul(self.W,self.x[n])
                                       + cp.array(sp.matmul(self.W_in, sp.hstack((sp.array([1]),u))))
                                       + cp.array(sp.matmul(self.W_fb, Y)))
        self.x[n+1] = cp.multiply((1-cp.array(self.alpha_matrix)), cp.array(self.x[n])) \
              + cp.multiply(cp.array(self.alpha_matrix), x_n_tilde)

    def calculate_W_out(self, Y_target, N_x, beta, train_start_timestep, train_end_timestep):
        # see Lukosevicius Practical ESN eqtn 11
        # Using ridge regression
        N_u = sp.shape(Y_target)[0]
        X = cp.array(cp.vstack((cp.ones((1,train_end_timestep-train_start_timestep)),
                                cp.array(Y_target[:,train_start_timestep:train_end_timestep]),
                                self.x[train_start_timestep:train_end_timestep].transpose())))
        # Ridge Regression
        self.W_out = cp.asnumpy(cp.matmul(cp.array(Y_target[:, train_start_timestep+1:train_end_timestep+1]),\
                                          cp.matmul(X.transpose(),
                                                    cp.array(cp.linalg.inv(
                                                        cp.add(cp.matmul(X,X.transpose()),beta*cp.identity(1+N_x+N_u)))))))

    def output_Y(self, u, n):
        one_by_one = sp.array([1])
        # Eqtn 4
        concatinated_matrix = sp.hstack((sp.hstack((one_by_one,u)),
                                         cp.asnumpy(self.x[n])))
        output_Y_return = sp.matmul(self.W_out, concatinated_matrix)
        return output_Y_return