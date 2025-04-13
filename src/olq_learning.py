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

class OLQ():
    """
    """
    def __init__(self, 
                 T: int=500,
                 eta: float=0.1,  # step size
                 exp_name: str='test',
                 nu: float=0.1,  # trace limit
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))

        self.T = T  # number of iterations
        self.eta = eta
        self.nu = nu
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

        self.traj = fcs.traj_initialization(SIM_PARAMS['dt'])
        self.env = fcs.env_initialization(SIM_PARAMS)
        
        self.E = torch.eye((self.l + self.l) ** 2, dtype=torch.float32).to(self.device) * 1e-2

        self.Q = torch.eye(self.l, dtype=torch.float32).to(self.device)
        self.R = torch.eye(self.l, dtype=torch.float32).to(self.device) * 0.05

        self.L = torch.cat([
            torch.cat([self.Q, -self.Q], dim=1),
            torch.cat([-self.Q, self.R], dim=1)
        ], dim=0)

        self.A = np.zeros((self.l, self.l))

    def update_L(self, K) -> Array2D:
        """
        """
        self.L[self.l:, self.l:] = K.t()@self.R@K
    
    @staticmethod
    def get_u(K: Array2D, yref: Array2D, V: Array2D) -> Array2D:
        """Return the input.
        """
        mean = K@yref.reshape(-1, 1)
        # return np.random.multivariate_normal(mean.flatten(), V)
        return mean

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

    @staticmethod
    def cal_K(Sigma, l):
        """
        """
        Sigma_rr = Sigma[l:, l:]
        Sigma_xr = Sigma[:l, l:]
        return Sigma_xr@torch.linalg.inv(Sigma_rr)
    
    @staticmethod
    def cal_V(Sigma, K, l):
        Sigma_xx = Sigma[:l, :l]
        Sigma_rr = Sigma[l:, l:]
        return Sigma_xx - K@Sigma_rr@K.t()

    def get_K_V(self, Sigma, l):
        K = self.cal_K(Sigma, l)
        V = self.cal_V(Sigma, K, l)
        return K, V
    
    def get_W(self, yout, yref, K):
        _yout = self.B@K@yref.reshape(-1, 1)
        dy = (yout.flatten() - _yout.flatten()) ** 2
        return np.diag(dy)

    def projection(self, Sigma_tilde, nu, W, K):
        """
        """  
        Sigma = cp.Variable((self.l * 2, self.l * 2), symmetric=True)
        
        M = np.hstack((self.A, self.B@K))
        
        objective = cp.Minimize(cp.norm(Sigma - Sigma_tilde, 'fro'))
        
        constraints = [
            Sigma >> 0,  
            cp.trace(Sigma) <= nu,  
            Sigma[:self.l, :self.l] == M @ Sigma @ M.T + W
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False) 
        
        return Sigma.value

    def learning(self) -> None:
        loss_list = []
        it_list = []
        Sigma = torch.eye(self.l*2, dtype=torch.float32).to(self.device)

        for i in range(self.T):
            yref = self.get_traj()
            K, V = self.get_K_V(Sigma, self.l)
            self.update_L(K)
            u = self.get_u(K.cpu().numpy(), yref, V.cpu().numpy())
            yout, _ = self.env.one_step(u.flatten())           
            W = self.get_W(yout.flatten(), yref.flatten(), K.cpu().numpy())
            Sigma = Sigma - self.eta * self.L
            Sigma = self.projection(Sigma.cpu().numpy(), self.nu, W, K.cpu().numpy())
            Sigma = torch.from_numpy(Sigma).float().to(self.device)
            loss = fcs.get_loss(yref.flatten(), yout.flatten())

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
            
        self.save_data(iteration=i,
                        loss=loss_list)
