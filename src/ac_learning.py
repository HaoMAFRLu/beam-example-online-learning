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

class AC():
    """
    """
    def __init__(self, 
                 T: int=500,
                 H: int=80,
                 eta: float=0.1,  # step size
                 exp_name: str='test',
                 kappa: float=5.0,  # trace limit
                 gamma: float=0.05,
                 sigma: float=0.1,
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))

        self.T = T  # number of iterations
        self.eta = eta
        self.H = H
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma

        self.root = fcs.get_parent_path(lvl=1)

        # folder_name = fcs.get_folder_name()
        folder_name = 'ac_H' + str(int(self.H))
        
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
        
        self.B = torch.from_numpy(self.B).float().to(self.device)
        # self.K = torch.ones_like(self.B).float().to(self.device) * 0.1
        self.K = torch.linalg.pinv(self.B)

        self.traj = fcs.traj_initialization(SIM_PARAMS['dt'])
        self.env = fcs.env_initialization(SIM_PARAMS)
        
        self.get_M_list()

    def get_M_list(self):
        """
        """
        self.M_list = []
        self.w_list = []
        for i in range(self.H):
            temp = self.sigma * torch.randn(self.l, self.l, requires_grad=True, dtype=torch.float32, device=self.device)
            self.M_list.append(temp.detach().clone().requires_grad_())
            self.w_list.append(torch.zeros((self.l, 1), dtype=torch.float32, device=self.device))
            
    def get_u(self, yref) -> Array2D:
        """Return the input.
        """
        y = torch.from_numpy(yref.reshape(-1, 1)).float().to(self.device)
        u = self.K@y
        for i in range(self.H): 
            u += self.M_list[i]@self.w_list[i]

        return u

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

    def get_grad(self, yout, yref, u):
        a = torch.from_numpy(yout.reshape(1, -1) - yref.reshape(1, -1)).float().to(self.device)
        phi = a @ self.B @ u
        phi.backward(retain_graph=True)
        self.M_grad = []
        for i in range(self.H):
            self.M_grad.append(self.M_list[i].grad.clone())
            self.M_list[i].grad.zero_() 

    def update_M(self):
        """
        """
        M_temp = [self.M_list[i] - self.eta * self.M_grad[i] for i in range(self.H)]
        for i in range(self.H):
            norm_bound = self.kappa**4 * (1 - self.gamma)
            # if torch.norm(M_temp[i]) > norm_bound:
            #     self.M_list[i] = (M_temp[i] * (norm_bound / torch.norm(M_temp[i]))).detach().clone().requires_grad_()
            # else:
            self.M_list[i] = M_temp[i].detach().clone().requires_grad_()

        temp = self.sigma * torch.randn(self.l, self.l, requires_grad=True, dtype=torch.float32, device=self.device)        
        self.M_list.insert(0, temp.detach().clone().requires_grad_()) 
        self.M_list.pop()

    def update_w(self, yout, u):
        """
        """
        y = torch.from_numpy(yout.reshape(-1, 1)).float().to(self.device)
        w = y - self.B@u
        # w = y
        self.w_list.insert(0, w) 
        self.w_list.pop()   

    def learning(self) -> None:
        loss_list = []
        it_list = []

        for i in range(self.T):
            yref = self.get_traj()
            u = self.get_u(yref)
            yout, _ = self.env.one_step(u.detach().cpu().numpy().flatten())
            loss = fcs.get_loss(yref.flatten(), yout.flatten())
            self.get_grad(yout, yref, u)

            self.update_M()
            # self.update_w(yout, u)
            self.update_w(yout, u)

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
            
        
