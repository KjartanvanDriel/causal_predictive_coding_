import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import numpy as np

import evaluate

plt.style.use("./dark_theme.mplstyle")

def plot_weights(weights, name = ""):
    N = np.shape(weights)[0]

    x = int(np.ceil(np.sqrt(N)))
    y = int(np.sqrt(N))

    fig, axes = plt.subplots(x,y)
    fig.suptitle(f"{name} weights")
    for i, (w_z, ax) in enumerate(zip(weights,axes.flatten())):
        im = ax.imshow(np.atleast_2d(w_z), cmap = 'RdBu', vmin=-1,vmax=1)
        ax.set_title("Neuron {}".format(i))
        ax.axis("off")

    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, fig.add_axes([0.85, 0.15, 0.05, 0.7]))

def plot_activity(z, dt = 0.2, group_name = ""):
    t = np.arange(np.shape(z)[1])*dt
    plt.figure()
    plt.suptitle(f"{group_name} - activity")
    for i, z_i in enumerate(z):
        plt.subplot(np.shape(z)[0] + 1,1, i+1)
        plt.plot(t, z_i)
        plt.ylim([0,2])
        plt.ylabel(rf"$z$")
    plt.xlabel(r"$t$ (ms)")

def plot_membrane_potential(z, dt = 0.2, group_name = ""):
    t = np.arange(np.shape(z)[1])*dt
    plt.figure()
    plt.suptitle(f"{group_name} - membrane potential")
    for i, z_i in enumerate(z):
        plt.subplot(np.shape(z)[0] + 1,1, i+1)
        plt.plot(t, z_i)
        plt.ylim([-1,1])
        plt.ylabel(rf"$u$")
    plt.xlabel(r"$t$ (ms)")

def plot_decoding(x_log, decoding_log, name = "", dt=0.2):
    t = np.arange(np.shape(x_log)[1])*dt

    plt.figure()
    plt.suptitle(f"{name} input/decoding")
    for i, (x, decoding) in enumerate(zip(x_log, decoding_log)):
        plt.subplot(np.shape(x_log)[0],1,i+1)
        plt.plot(t ,decoding, label=r'$\hat{x}$')
        plt.plot(t, x, label='x')
        plt.legend()
    plt.xlabel(r"$t$ (ms)")

def plot_spike_trace(s_log, name= "", dt=0.2,ylabel="y"):
    t = np.arange(np.shape(s_log)[1])*dt

    plt.figure()
    plt.suptitle(f"{name} spikes")
    for i, s in enumerate(s_log):
        plt.subplot(np.shape(s_log)[0],1,i+1)
        times = t[np.where(s == 1)]
        #ugly but efficient plotting
        plt.plot(np.repeat([times],2,axis=0), np.repeat([[0,1]], len(times),axis=0).T,color='#9A72AC')
        plt.ylabel(rf"${ylabel}$")

def plot_rate(s_log, name ="", dt=0.2):
    window_length = 1000

    plt.figure()
    plt.suptitle("{name} rate")
    for spikes in s_log:
        rate = evaluate.rolling_average(spikes, window_length)

        t = (np.arange(len(rate)) + int(window_length))*dt
        plt.plot(t, rate)

        plt.ylabel(r"$r$")

    plt.xlabel(r"$t$ (ms)")


def plot_decoding_performance(inputs):
    window_length = 10

    rv = False

    plt.figure()
    plt.suptitle("Decoding performance")
    for inp in inputs:
        if inp.log:
            decoding_performance = evaluate.rolling_decoding_performance(inp.x_log - inp.decoding_log)

            t = (np.arange(len(decoding_performance)) + int(window_length/2))*inp.dt
            plt.plot(t, decoding_performance, label=inp.identifier)

            rv = True
    if rv:
        plt.legend()
        plt.xlabel(r"$t$ (ms)")
        plt.ylabel(r"$L$")

    return rv

def plot_threshholds(T, dt, name=""):
    plt.figure()
    plt.suptitle(f"{name} threshholds $T_j$")
    t = np.arange(np.shape(T)[1])*dt
    plt.plot(t,T.T)
    plt.xlabel(r"$t$ (ms)")
    plt.ylabel(r"$T_j$")


def plot_group_log(group, group_name = ""):
    plot_activity(np.array(group.z_log))
    plt.suptitle(f"{group_name} activity")

    plot_activity(np.array(group.u_log))
    plt.suptitle(f"{group_name} membrane potential")


def plot_mouse_experiment(model):
    groups, inputs, connections = model

    V1 = groups['V1']
    M2 = groups['M2']

    center_optic_flow = inputs['Visual'].x_log[1]


    t = np.arange(np.shape(center_optic_flow)[0])*V1.dt

    plt.figure(figsize=(12,12))
    #plt.suptitle(f"{group_name} - membrane potential")
    for i, (u_i, s_i) in enumerate(zip(V1.u_log,V1.s_log)):
        plt.subplot(18,1, i+1)
        plt.plot(t, u_i)

        times = t[np.where(s_i == 1)]
        potential = u_i[np.where(s_i ==1)]

        plt.plot(np.repeat([times],2,axis=0), np.repeat([[-0.5,1.5]], len(times),axis=0).T + potential,color='#9A72AC')
        plt.ylim([-1,2.5])
        plt.axis('off')

    for i, (u_i, s_i) in enumerate(zip(M2.u_log,M2.s_log)):
        plt.subplot(18,1, 14+i)
        plt.plot(t, u_i)

        times = t[np.where(s_i == 1)]
        potential = u_i[np.where(s_i ==1)]

        plt.plot(np.repeat([times],2,axis=0), np.repeat([[-0.5,1.5]], len(times),axis=0).T + potential,color='#9A72AC')
        plt.ylim([-1,2.5])
        plt.axis('off')

    plt.subplot(18,1, (17,18))
    plt.plot(t,center_optic_flow)
    plt.ylim([0,2])
    plt.axis('off')


    plt.subplots_adjust(right=0.9)

    #plt.annotate("", xy=(0.9,1/3), xycoords='figure fraction', xytext=(0.9,1), textcoords='figure fraction',arrowprops=dict(connectionstyle="arc",angleA=0,angleB=0,armA=None,armB=None, rad=0.0))

    #plt.annotate("V1",xy=(0.92,2/3),xycoords='figure fraction', xytext=(0.92,2/3), textcoords='figure fraction', fontsize=30, arrowprops=dict(arrowstyle="->",
                            #connectionstyle="arc3"))

