import numpy as np

from numba import jit

class SB_network():

    pass

class DB_network():
    def __init__(self, N, input_size, x,
                 dt = 0.2 ,    tau = 10,    deltaU = 1/0.05,
                 rho = 15,   lambd = 0.03,   eta_T = 3e-4,
                 eta_F = 3e-4,    eta_W = 3.0e-5, eta_D = 1.0e-6, **kwargs):

        self.N = N
        self.dt = dt                                    #mean spike timewindow
        self.tau = tau                                  #(decay) timescale
        self.deltaU = deltaU                            #Probability scaling
        self.rho = rho * 1e-3                           #mean spikes per second => spike per timewindow
        self.lambd = lambd                              #weight decay parameters
        self.eta_T = eta_T*dt                           #Spike rate learning rate
        self.eta_F = eta_F*dt                           #Feedforward learning rate
        self.eta_W = eta_W*dt                           #Recurrent learning rate
        self.eta_D = eta_D*dt                           #Decoder learning rate

        self.log = False
        self.log_z = []
        self.log_s = np.zeros(N)
        self.log_u = []
        self.log_x_hat = []


        self.decay = np.exp(-dt/tau)

        self.timescale = self.tau
        self.current_time=0

        self.input_size = input_size
        self.num_of_neurons = N

        self.x = x                                      #Input vectors

        self.z = np.zeros(N)                            #Postsynaptic
        self.u = np.zeros(N)                            #Presynaptic
        self.u_dendrites = np.zeros((input_size, N))    #Dendritic compartments
        self.s = np.zeros(N)                            #Spike train
        self.s_history = [np.zeros(N)]

        self.T = np.ones(N)

        #Feedforward weights
        #self.F = kwargs.get("feedforward_weights",np.ones((N,input_size))*0.02)
        self.F = kwargs.get("feedforward_weights",np.random.randn(N,input_size)*0.02)

        #Decoder weights
        self.D = kwargs.get("decoder_weights",np.ones((input_size,N))*0)

        #Recurrent weights
        self.W = kwargs.get("recurrent_weights",np.ones((input_size,N, N))*0)
        #self.W = kwargs.get("recurrent_weights",np.random.rand(input_size,N, N)*2 -1)

        #Inhibitory (I) coupling
        self.I_coupling = np.zeros((input_size,N))

    def update_z(self):
        #s has not been updated yet so it is from the last frame
        self.z = (self.z + self.s)*self.decay

    def update_u_dendrites(self):
        self.u_dendrites = (self.x * self.F).T + np.inner(self.W, self.z) + self.I_coupling

    def update_u(self):
        self.u = np.sum(self.u_dendrites, axis=0)
        #self.u = self.F.dot( self.x - self.F.T.dot(self.z) )

    def update_s(self):
        random_uniform = np.random.rand(self.num_of_neurons)
        #print(random_uniform)
        #print(-(self.u - self.T)/self.deltaU)
        #print(-self.u)
        #print(self.T)
        self.s = (1/(1+np.exp(-(self.u - self.T)/self.deltaU)) > random_uniform)*1

    def update_T(self):
        self.T += self.eta_T*(self.s - self.rho * self.dt)

    def update_F(self):

        #self.F += self.eta_F*(np.einsum('ij,i,ji',1/self.F,self.z,self.u_dendrites)- self.lambd * self.F )
        self.F += self.eta_F*np.einsum('i,j',self.z, self.x - np.inner(self.F.T,self.z))
        #self.F += self.eta_F*(np.einsum('i,j',self.z, self.x) - np.einsum('i,ij', self.z**2, self.F))

    def update_D(self):
        self.D += self.eta_D*np.einsum('j,i', self.z, self.x - np.inner(self.D,self.z)) -0* self.lambd * self.D

    def update_W(self):
        self.W += -self.eta_W *(np.einsum('k,ij',self.z,self.u_dendrites) - 0*self.lambd*self.W)

    def print_input(self):
        print(self.x)

    def decode_F(self):
        return np.inner(self.F.T,self.z)

    def decode_D(self):
        return np.inner(self.D,self.z)

    def update_network_dynamics(self):
        self.update_z()
        self.update_u_dendrites()
        self.update_u()
        self.update_s()


        #Reset inhibitory coupling
        self.I_coupling *= 0

        self.current_time += 1

        self.log_s+= self.s

        #Logging
        if self.log:
            #self.log_s.append(self.s)
            self.log_z.append(self.z)
            self.log_u.append(self.u)
            self.log_x_hat.append(self.decode_F())

    def update_network_parameters(self):
        self.update_T()

        self.update_F()
        self.update_W()
        self.update_D()

    def update_network_parameters_analytic(self):
        self.F = self.D
        self.W = np.einsum('')

        self.update_T()
        self.update_D()


    @jit
    #Legacy idea
    def optimized_update_network_dynamics(self):
        self.z = (self.z + self.s)*self.decay
        self.u_dendrites = (self.x * self.F).T + np.inner(self.W, self.z)
        self.u = np.sum(self.u_dendrites, axis=0)
        self.s = (1/(1+np.exp(-(self.u - self.T)/self.deltaU)) > np.random.rand(self.num_of_neurons))*1
        self.T += self.eta_T*(self.s - self.rho * self.delta)

        self.F += self.eta_F*np.einsum('i,j',self.z, self.x - np.inner(self.F.T,self.z))
        self.W += -self.eta_W *(np.einsum('k,ij',self.z,self.u_dendrites) - 0*self.lambd*self.W)


class DB_I_coupling():

    def __init__(self, A, B, eta_Z = 3.0e-5, **kwargs):

        self.eta_Z = eta_Z
        self.lambd = 0

        self.A = A
        self.B = B

        #Analog of lateral weights W
        self.Z = kwargs.get("weights",np.ones((B.input_size,B.num_of_neurons,A.num_of_neurons)))

        #Coupling to be send to B
        self.I_u = np.ones((B.input_size, B.num_of_neurons))*0

    def update_Z(self):
        self.Z += -self.eta_Z *(np.einsum('k,ij',self.A.z,self.B.u_dendrites) - 0*self.lambd*self.Z)

    def update_I_u(self):
        self.I_u = np.inner(self.Z, self.A.z)
        self.B.I_coupling += self.I_u

    def update_coupling_dynamics(self):
        self.update_I_u()

    def update_coupling_parameters(self):
        self.update_Z()



