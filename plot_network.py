import numpy as np

import matplotlib.pyplot as plt

def plot_spikes(s,dt = 0.2):

    time_stamp, neuron= np.where(s>0)

    plt.figure()
    plt.scatter(time_stamp*dt, neuron + 1)

def plot_activity(z, dt = 0.2):
    t = np.arange(len(z))*dt
    plt.figure()
    for n in range(np.shape(z)[1]):
        plt.subplot(np.shape(z)[1] + 1,1, n+1)
        plt.plot(t, z[:,n])


def plot_decoding(x,x_hat, dt = 0.2):

    t = np.arange(len(x))*dt

    plt.figure()

    if len(np.shape(x)) == 1:
        plt.plot(t,x_hat, label=r'$\hat{x}$')
        plt.plot(t,x, label=r'$x$')
    else:
        for y_i in range(np.shape(x)[1]):
            plt.subplot(np.shape(x)[1],1, y_i + 1)
            plt.plot(t,x_hat[:,y_i], label=r'$\hat{x}$')
            plt.plot(t,x[:,y_i], label=r'$x$')
            plt.plot(t,x[:,y_i] - x_hat[:,y_i], label=r'$\Delta x$')

    plt.legend()



