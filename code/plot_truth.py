import matplotlib.pyplot as plt
import numpy as np


def plot_truth(states, data):
    Nscans = states.shape[0]

    plt.plot(np.arange(Nscans), states[:,0], 'r' ,label='True state')
    plt.plot(np.arange(Nscans), data.reshape(-1), 'xb', label = 'Observations' )
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.savefig('True2')
    # plt.show()