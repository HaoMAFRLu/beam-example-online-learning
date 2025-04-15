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
        folder_name = fcs.get_folder_name()
        
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
    
    def get_l_loss(self, y1: Array, y2: Array) -> float:
        """
        """
        dy = torch.from_numpy(y1 - y2).float().to(self.device)
        return torch.dot(dy, dy)

    def run_mult_rounds(self, K: Array2D) -> None:
        """
        """
        for i in range(self.Ti):
            self.cur_it += 1

            yref = self.get_traj()
            u = self.get_u(K, yref)
            yout, _ = self.env.one_step(u.flatten())

            self.us[:, i] = torch.from_numpy(u.flatten()).float().to(self.device)
            self.ys[:, i] = torch.from_numpy(yout.flatten()).float().to(self.device)
    
            loss = fcs.get_loss(yref.flatten(), yout.flatten())

            self.save_data(iteration=self.cur_it,
                           loss=loss,
                           yref=yref,
                           yout=yout)
            
            fcs.print_info(Iteration=[str(self.cur_it)],
                           Loss=[loss])
            
            self.losses.append(loss)
            self.iterations.append(self.cur_it)

            if self.is_vis is True:
                plt.clf()
                plt.plot(self.iterations, self.losses, label='Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Process')
                plt.legend()
                
                plt.pause(0.01)
                
                time.sleep(0.01)
    
    def update_B(self, Y, U):
        """
        """
        return Y@U.t()@torch.linalg.inv(U@U.t())
    
    def save_data(self, iteration: int,
                  **kwargs) -> None:
        """Save the data
        """
        file_name = str(iteration)
        path_data = os.path.join(self.path_data, file_name)
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)

    def learning(self) -> None:

        for i in range(self.Ts):
            self.run_mult_rounds(self.K)
            self.B = self.update_B(self.ys, self.us)
            self.K = self.policy_update(self.B)
            
        
