"""Implementation of MFLQ algorithm.
"""
import torch
import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List
import math
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
import cvxpy as cp

random.seed(10086)

import utils as fcs 
from mytypes import Array, Array2D, Array3D

class LINEAR():
    """
    """
    def __init__(self, 
                 T: int=500,
                 exp_name: str='test',
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))

        self.T = T  # number of iterations
        self.root = fcs.get_parent_path(lvl=1)
        folder_name = fcs.get_folder_name()
        
        self.path_model = os.path.join(self.root, 'data', exp_name, folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        self.initialization()
        
        self.is_vis = is_vis
        if self.is_vis is True:
            plt.ion()          

    @staticmethod
    def load_dynamic_model(l):
        B = fcs.load_dynamic_model()
        return B[:l, :l]
    
    def initialization(self) -> None:
        """
        """
        SIM_PARAMS, DATA_PARAMS, _ = fcs.get_params(self.root)
        self.l = math.floor(float(SIM_PARAMS['StopTime']) / SIM_PARAMS['dt'])
        self.B = self.load_dynamic_model(self.l)
        self.K = np.linalg.pinv(self.B)
        self.traj = fcs.traj_initialization(SIM_PARAMS['dt'])
        self.env = fcs.env_initialization(SIM_PARAMS)
        

    def get_u(self, K, yref) -> Array2D:
        """Return the input.
        """
        return K@yref.reshape(-1, 1)

    def get_traj(self):
        """Remove the first element.
        """
        yref, _ = self.traj.get_traj()
        return yref[0, 1:]

    def save_data(self,
                  iteration: int, 
                  **kwargs) -> None:
        """Save the data
        """
        file_name = str(iteration)
        path_data = os.path.join(self.path_data, file_name)
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)


    def learning(self) -> None:
        loss_list = []
        it_list = []

        for i in range(self.T):
            yref = self.get_traj()
            u = self.get_u(self.K, yref)
            yout, _ = self.env.one_step(u.flatten())
            loss = fcs.get_loss(yref.flatten(), yout.flatten())

            self.save_data(iteration=i,
                           loss=loss,
                           yref=yref,
                           yout=yout)
            
            fcs.print_info(
                Epoch=[str(i)],
                Loss=[loss])
            
            loss_list.append(loss)
            it_list.append(i)

            if self.is_vis is True:
                plt.clf()
                plt.plot(it_list, loss_list, label='Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Process')
                plt.legend()
                plt.pause(0.01)
                time.sleep(0.01)
            
        
