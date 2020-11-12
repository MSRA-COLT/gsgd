import torch
import numpy as np
from models import SimpleNet,  SimpleNet_cifar10
from gopt import gSGD
import matplotlib
import copy
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from dataset import simulation
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib import animation
from IPython.display import HTML

from itertools import zip_longest


def plot_3d(x_idx , y_idx, z_idx, rec_data=rec_sgd , num_picture=200, dir='sgd/' ):
    fig = plt.figure()
    ax = Axes3D(fig)
    # #
    # # X = np.arange(-4, 4, 0.25)
    # # Y = np.arange(-4, 4, 0.25)
    # # X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
    # # R = np.sqrt(X ** 2 + Y ** 2)
    # # # height value
    # # Z = np.sin(R)
    #
    #
    #
    num_data = len(rec_data)
    num_picture = num_picture
    plot_idx = np.linspace(0,num_data,num_picture)
    plot_idx = [int(x) for x in plot_idx]
    #
    x = rec_data[:,x_idx]
    y = rec_data[:, y_idx]
    z = rec_data[:,z_idx]
    #
    x_range = [np.min(x) , np.max(x)]
    y_range = [np.min(y) , np.max(y)]
    z_range = [np.min(z) , np.max(z)]

    for ii in range(num_picture):
        fig = plt.figure()
        ax = Axes3D(fig)
        print(ii)
        ax.plot(x[:plot_idx[ii ]], y[:plot_idx[ii]], z[:plot_idx[ii]] ,'-o' ,linewidth=1, markersize=2  )

        # ax.plot(x[:plot_idx[ii ]], [0], [0.3], '-o', linewidth=1, markersize=2)
        ax.set_xlim3d(x_range[0], x_range[1])
        ax.set_ylim3d(y_range[0], y_range[1])
        ax.set_zlim3d(z_range[0], z_range[1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=45., azim=90)
        # fig.show()
    #
        # fig.show()
        fig.savefig( dir +  str(ii)+'.png')
        plt.close()