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

class BA():
    """
    """
    def __init__(self, 
                 Ts: int=20,
                 Ti: int=49,
                 eta: float=0.1,
                 sigma: float=0.5,
                 exp_name: str='test',
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))
        self.root = fcs.get_parent_path(lvl=1)

        # folder_name = fcs.get_folder_name()
        
        self.Ts = Ts
        self.Ti = Ti
        self.eta = eta
        self.sigma = sigma

        folder_name = 'ba' + '_' + str(int(self.eta)) + '_' + str(int(self.Ti))
        self.path_model = os.path.join(self.root, 'data', exp_name, folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        
        self.initialization()
        
        self.loss_list = []
        self.it_list = []
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
        self.traj = fcs.traj_initialization(SIM_PARAMS['dt'])
        self.env = fcs.env_initialization(SIM_PARAMS)
        
        self.l = math.floor(float(SIM_PARAMS['StopTime']) / SIM_PARAMS['dt'])
        self.B = self.load_dynamic_model(self.l)
        self.B = torch.from_numpy(self.B).float().to(self.device)
        tmp = torch.linalg.pinv(self.B).float().to(self.device)
        # tmp = torch.ones_like(self.B).float().to(self.device) * 1.0
        self.K = tmp.detach().clone().requires_grad_()

        self.dus = torch.zeros((self.l, self.Ti), dtype=float).to(self.device)
        self.dys = torch.zeros((self.l, self.Ti), dtype=float).to(self.device)
        
    def get_u(self, K, yref) -> Array2D:
        """Return the input.
        """
        y = torch.from_numpy(yref.reshape(-1, 1)).float().to(self.device)
        return K@y

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

    def update_policy(self, yout, yref, u, G):
        a = yout.reshape(1, -1) - yref.reshape(1, -1)
        phi = a @ G @ u
        phi.backward(retain_graph=True)
        K_tmp = self.K - self.eta * self.K.grad
        self.K = K_tmp.detach().clone().requires_grad_()
 
    def to_torch(self, a):
        return torch.from_numpy(a.flatten()).float().to(self.device)
    
    def run_nominal(self, K):
        """
        """
        yref = self.get_traj()
        u = self.get_u(K, yref)
        yout, _ = self.env.one_step(u.detach().cpu().numpy().flatten())
        return self.to_torch(yref), self.to_torch(yout), u.flatten() 

    def get_random_u(self, u):
        noise = np.random.randn(*u.shape) * self.sigma
        return u + noise

    def run_random(self, u, yout):
        u_bar = self.get_random_u(u.flatten())
        y_bar, _ = self.env.one_step(u_bar)
        # return y_bar.flatten()-yout, u_bar - u
        return y_bar.flatten(), u_bar

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

    def learning(self) -> None:
        for i in range(self.Ts):
            yref, yout, u = self.run_nominal(self.K)
            self.save_and_print(i,
                                yout.detach().cpu().numpy().flatten(),
                                yref.detach().cpu().numpy().flatten())

            for ii in range(self.Ti):
                dy, du = self.run_random(u.detach().cpu().numpy().flatten(),
                                         yout.detach().cpu().numpy().flatten())
                self.dus[:, ii] = self.to_torch(du)
                self.dys[:, ii] = self.to_torch(dy)

            G = (self.dys@torch.linalg.pinv(self.dus)).float()
            self.update_policy(yout, yref, u, G)

            
            
            
            
            
            
        
