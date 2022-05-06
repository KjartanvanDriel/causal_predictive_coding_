import numpy as np
from tqdm import tqdm
from network import DB_network, DB_I_coupling
from plot_network import plot_spikes, plot_decoding, plot_activity

import os
import prepare_input
import pickle
import matplotlib.pyplot as plt

err_func = lambda x : 1/(2*len(x))*np.inner(x,x)

learn = True

""" Recreating the simple mouse experiment """

egomotion_input, visual_input = prepare_input.visual_egomotion_input(1000,6)


dt = 0.2 #[ms]/[step]

inp_func_egomotion = lambda t : prepare_input.visual_egomotion_input_func(egomotion_input, t, presentation_time = int(100/dt), fade_time=int(10/dt))
inp_func_visual = lambda t : prepare_input.visual_egomotion_input_func(visual_input, t, presentation_time = int(100/dt), fade_time=int(10/dt)) # i.e. 100 ms

#inp_func = lambda t : [(np.round(np.sin(t/1000) +1/2)), np.round((np.sin(t/1000+np.pi)+ 1)/2)]
inp_egomotion = inp_func_egomotion(0)
inp_visual = inp_func_visual(0)


if learn:
    T = 50* int( 1000 / dt )
else:
    T = 5 * int( 1000 / dt )

errorlog=[]

#TODO
#feedforward_weights = np.exp(np.maximum(0,0.3*np.random.randn(9,16**2)-0.2)) - 0.99

if os.path.exists("network_pickle_global"):
    M2_network, V1_network, M2_V1_I_coupling , F_V_M2 = pickle.load(open("network_pickle_global", "rb"))
    M2_network.x = inp_egomotion
    V1_network.x = inp_visual
    M2_network.log_s *= 0
    V1_network.log_s *= 0

else:
    M2_network = DB_network(2, 2, inp_egomotion, rho=15, deltaU=0.1,eta_T=3e-2) #1/0.1)
    V1_network = DB_network(12, 6, inp_visual, rho=8, deltaU= 0.05)

    M2_V1_I_coupling = DB_I_coupling(M2_network, V1_network)

    F_V_M2 = (np.exp(np.maximum(0,0.3*np.random.randn(2,6)-0.2)) - 0.99)


#M2_network.eta_F = M2_network.dt * 3e-4
V1_network.eta_T = V1_network.dt * 3e-1

if not learn:
    M2_network.log =True
    V1_network.log =True

    V_x_hat_log = []

for t in tqdm(range(T)):
    inp_egomotion[:] = inp_func_egomotion(t)
    inp_visual[:] = inp_func_visual(t)

    M2_V1_I_coupling.update_coupling_dynamics()
    M2_network.update_network_dynamics()
    V1_network.update_network_dynamics()

    #learning
    #M2_V1_I_coupling.update_coupling_parameters()
    #M2_network.update_network_parameters()
    #V1_network.update_network_parameters()

    errorlog.append([
        #err_func(inp_visual - np.inner(V1_network.F.T,V1_network.z) + np.inner(F_V_M2.T, M2_network.z)),
        err_func(inp_visual - np.inner(V1_network.F.T,V1_network.z) + np.inner(F_V_M2.T, M2_network.z)),
        err_func(inp_egomotion - np.inner(M2_network.F.T,M2_network.z))
    ])

    if learn:
        #       Analytic learning
        M2_network.update_T()
        V1_network.update_T()
        # First feedforward learning
        M2_network.update_F()
        V1_network.update_F()
        F_V_M2 += M2_network.eta_F*np.einsum('i,j',M2_network.z, inp_visual - np.inner(F_V_M2.T,M2_network.z))

        #Lateral weights learning
        M2_network.W = - np.einsum('ji,ki->ijk',M2_network.F ,M2_network.F)
        V1_network.W = - np.einsum('ji,ki->ijk',V1_network.F, V1_network.F)
        M2_V1_I_coupling.Z = - np.einsum('ji,ki->ijk',V1_network.F, F_V_M2)

    else:
        V_x_hat_log.append(np.inner(V1_network.F.T,V1_network.z) + np.inner(F_V_M2.T, M2_network.z))
        pass

print(M2_network.T)
print(V1_network.T)
print(M2_network.F)
print(V1_network.F)
print(M2_V1_I_coupling.Z)
#M2_V1_I_coupling.clean()
#M2_network.clean()
#V1_network.clean()


#Saving the network

plt.figure()
plt.plot(errorlog)

if learn:
    print((M2_network.log_s/T)/dt*1e3)
    print((V1_network.log_s/T)/dt*1e3)
    pickle.dump((M2_network,V1_network,M2_V1_I_coupling, F_V_M2), open("network_pickle_global","wb"))
    print("pickled")
    plt.show()
else:
#if learn or not learn:
    skip = 0

    plot_activity(np.array(M2_network.log_z))
    plt.suptitle("M2 activity")
    plot_activity(np.array(V1_network.log_z))
    plt.suptitle("V1 activity")

    plot_activity(np.array(M2_network.log_u))
    plt.suptitle("M2 membrane potential")
    plot_activity(np.array(V1_network.log_u))
    plt.suptitle("V1 membrane potential")

    plot_decoding(inp_func_egomotion(np.arange(len(M2_network.log_x_hat)))[-skip:], np.array(M2_network.log_x_hat)[-skip:] )
    plt.title("M2 decoding")
    plot_decoding(inp_func_visual(np.arange(len(V_x_hat_log)))[-skip:], np.array(V_x_hat_log)[-skip:] )
    plt.title("V1 decoding")

    plt.show()

