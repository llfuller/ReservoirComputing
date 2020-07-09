import scipy as sp
import scipy.sparse
import cupy as cp
import os.path

"""Build an Echo State Network"""
class ESN: # TODO: Speed this up! Parallelize or use sparse?
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

    def __init__(self, N_x, N_u, N_y, sparsity,
                 x_initial, alpha_input, scaling_W,
                 scaling_W_in, scaling_W_fb,
                 train_end_timestep, timesteps_for_prediction):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x)
        # alpha_input has shape (N_x)
        self.W = self.build_W(N_x, sparsity, scaling_W)#input_W
        self.W_in = self.build_W_in(N_x, N_u, scaling_W_in)#input_W_in
        self.W_fb = self.build_W_fb(N_x, N_u, scaling_W_fb)#input_W_fb
        self.x = sp.zeros((train_end_timestep+ timesteps_for_prediction, N_x))
        self.x[0] = x_initial
        self.alpha_matrix = alpha_input

    def build_W(self, N_x, sparsity, scaling_W):
        # N_x integer
        # sparsity between 0 and 1 inclusive
        # scaling_W >= 1
        if os.path.isfile('./W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt'):
            W = sp.loadtxt('./W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt')
        else:
            # Build sparse adjacency matrix W:
            W_unnormalized = sp.multiply(sp.random.choice((-1,1), size=(N_x,N_x)),
                                         sp.sparse.random(N_x,N_x, density = sparsity).todense())
            # Normalize by largest eigenvalue and additional scaling factor
            # to control decrease of spectral radius.
            spectral_radius = sp.amax(abs(sp.linalg.eigvals(W_unnormalized)))
            print("SPECTRAL RADIUS IS IS "+str(spectral_radius))
            W = sp.float32(sp.multiply( scaling_W, sp.divide(W_unnormalized,spectral_radius) ))
            sp.savetxt('W_(adjacency)/W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt',W, fmt = '%.4f')
        return W

    def build_W_in(self, N_x, N_u, scaling_W_in):
        W_in = (1.0/scaling_W_in)*sp.random.random((N_x, 1+N_u))
        return W_in

    def build_W_fb(self, N_x, N_u, scaling_W_fb):
        W_fb = (1.0/scaling_W_fb)*sp.random.random((N_x, N_u))
        return W_fb

    def update_reservoir(self, u, n, Y):
        # u is input at specific time
        #   u has shape (N_u (3 for L63))
        # See page 16 eqtn 18 of Lukosevicius PracticalESN for feedback info.
        x_n_tilde = sp.tanh(sp.matmul(self.W,self.x[n])
                            + sp.matmul(self.W_in, sp.hstack((sp.array([1]),u)))
                            + sp.matmul(self.W_fb, Y))
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
        # plt.figure()
        # plt.plot(X)
        # plt.show()
        self.W_out = W_out

    def output_Y(self, u, n):
        one_by_one = sp.array([1])
        # concatinated_matrix has dimension (231,)
        # Eqtn 4
        concatinated_matrix = sp.hstack((sp.hstack((one_by_one,
                                                    u)),
                                         self.x[n]))
        return sp.matmul(self.W_out, concatinated_matrix)