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
import cvxpy as cp

random.seed(10086)

import utils as fcs 
from mytypes import Array, Array2D, Array3D

class RAC():
    """
    """
    def __init__(self, 
                 Ts: int,
                 Ti: int,
                 exp_name: str,
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))
        self.root = fcs.get_parent_path(lvl=1)
        # folder_name = fcs.get_folder_name()
        
        folder_name = 'rac'+  str(int(Ti))
        self.path_model = os.path.join(self.root, 'data', exp_name, folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        self.Ts = Ts
        self.Ti = Ti

        self.initialization()
        
        self.is_vis = is_vis
        self.losses = []
        self.iterations = []

        self.cur_it = -1

        self.loss_list = []
        self.it_list = []

        if self.is_vis is True:
            plt.ion()          

    def policy_update(self, B) -> Array2D:
        """Initialize the policy.
        """
        return torch.linalg.pinv(B).cpu().numpy()

    @staticmethod
    def load_dynamic_model(l):
        B = fcs.load_dynamic_model()
        return B[:l, :l]
    
    def initialization(self) -> None:
        """
        """
        SIM_PARAMS, DATA_PARAMS, _ = fcs.get_params(self.root)
        self.l = math.floor(float(SIM_PARAMS['StopTime']) / SIM_PARAMS['dt'])
        self.traj = fcs.traj_initialization(SIM_PARAMS['dt'])
        self.env = fcs.env_initialization(SIM_PARAMS)

        self.B = self.load_dynamic_model(self.l)
        self.B = torch.from_numpy(self.B).float().to(self.device)
        self.K = self.policy_update(self.B)

        self.ys = torch.zeros((self.l, self.Ti), dtype=float).to(self.device)
        self.us = torch.zeros((self.l, self.Ti), dtype=float).to(self.device)

    @staticmethod
    def get_u(K: Array2D, yref: Array2D) -> Array2D:
        """Return the input.
        """
        return K@yref.reshape(-1, 1)

    def get_traj(self):
        """Remove the first element.
        """
        yref, _ = self.traj.get_traj()
        return yref[0, 1:]
    
    def update_B(self, Y, U):
        """
        """
        return Y@torch.linalg.pinv(U)
    
    def save_data(self, iteration: int,
                  **kwargs) -> None:
        """Save the data
        """
        file_name = str(iteration)
        path_data = os.path.join(self.path_data, file_name)
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)

    def save_and_print(self, it, yout, yref):
        loss = fcs.get_loss(yref.flatten(), yout.flatten())
        self.save_data(iteration=it,
                        loss=loss,
                        yref=yref,
                        yout=yout)
        fcs.print_info(
                Epoch=[str(it)],
                Loss=[loss])
        
        self.loss_list.append(loss)
        self.it_list.append(it)

        if self.is_vis is True:
            
            plt.clf()
            plt.plot(self.it_list, self.loss_list, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Process')
            plt.legend()
            
            plt.pause(0.01)
            
            time.sleep(0.01)
    
    def run_nominal(self, K):
        """
        """
        yref = self.get_traj()
        u = self.get_u(K, yref)
        yout, _ = self.env.one_step(u.flatten())
        return yref.flatten(), yout.flatten(), u.flatten()
    
    def to_torch(self, a):
        return torch.from_numpy(a.flatten()).float().to(self.device)
    
    def learning(self) -> None:
        for i in range(self.Ts):
            yref, yout, u = self.run_nominal(self.K)
            self.save_and_print(i, yout, yref)
            
            for ii in range(self.Ti):
                yref, yout, u = self.run_nominal(self.K)
                self.us[:, ii] = self.to_torch(u)
                self.ys[:, ii] = self.to_torch(yout)

            self.B = self.update_B(self.ys, self.us)
            self.K = self.policy_update(self.B)
            
        
