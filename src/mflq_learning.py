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

random.seed(10086)

import utils as fcs 
from mytypes import Array, Array2D, Array3D

class MFLQ():
    """
    """
    def __init__(self, 
                 mode: str='v1',
                 T: int=100,
                 xi: float=0.1,
                 exp_name: str='test',
                 _w: float=0.1,
                 _sigma_a:float=0.1,
                 is_vis: bool=False):
        """Learning using model-free control of LQ systems.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))

        self.mode = mode
        self.T = T
        self.xi = xi
        self.root = fcs.get_parent_path(lvl=1)
        self._w = _w
        self._sigma_a = _sigma_a

        folder_name = fcs.get_folder_name()
        
        self.path_model = os.path.join(self.root, 'data', exp_name, folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        self.initialization()
        
        self.is_vis = is_vis
        self.losses = []
        self.iterations = []

        self.cur_it = -1

        if self.is_vis is True:
            plt.ion()          

    def policy_initialization(self) -> Array2D:
        """Initialize the policy.
        """
        return np.linalg.pinv(self.B)

    @staticmethod
    def get_constant(T: int, xi: float, mode: str) -> Tuple[int, int, int]:
        """Return some necessary constant.
        """
        if mode == 'v1':
            S = 40 # math.floor(math.pow(T, 1/3 - xi) - 1)
            Ts = 4
            Tv = 25 # math.floor(math.pow(T, 2/3 + xi))
        elif mode == 'v2':
            S = 40
            Ts = 4
            Tv = 25
        return S, Ts, Tv

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
        self.K = self.policy_initialization()
        self.S, self.Ts, self.Tv = self.get_constant(self.T, self.xi, self.mode)

        self.w = torch.eye(self.l, dtype=torch.float32).to(self.device) * self._w
        self.E = torch.eye((self.l + self.l) ** 2, dtype=torch.float32).to(self.device) * 1e-2

        self.Sigma_a = np.eye(self.l) * self._sigma_a

        self.Phi = torch.zeros((self.Tv + 1, self.l ** 2), dtype=torch.float32).to(self.device)
        # self.Phi_plus = torch.zeros((self.Tv, self.l ** 2), dtype=torch.float32).to(self.device)
        self.W = torch.zeros((self.Tv, self.l ** 2), dtype=torch.float32).to(self.device)
        self.c = torch.zeros((self.Tv, 1), dtype=torch.float32).to(self.device)

        self.tau_s = math.floor(self.Tv / self.Ts)
        self.ZPsi = torch.zeros((self.tau_s, (self.l + self.l) ** 2), dtype=torch.float32).to(self.device)
        self.ZPhi_plus = torch.zeros((self.tau_s + 1, self.l ** 2), dtype=torch.float32).to(self.device)
        self.ZW = torch.zeros((self.tau_s, self.l ** 2), dtype=torch.float32).to(self.device)
        self.Zc = torch.zeros((self.tau_s, 1), dtype=torch.float32).to(self.device)

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
    
    @staticmethod
    def get_random_traj(Sigma: Array2D) -> Array:
        """
        """
        mu = np.zeros(Sigma.shape[0])
        return np.random.multivariate_normal(mu, Sigma)

    def collect_data(self, K: Array2D,
                     Sigma: Array2D):
        """
        """
        for i in range(self.tau_s):
            yref = self.get_random_traj(Sigma)
            a = self.get_u(K, yref)
            yout, _ = self.env.one_step(a.flatten())

            self.Zc[i] = self.get_l_loss(yref.flatten(), yout.flatten())
            self.ZW[i, :] = self.w.flatten()   
            self.ZPhi_plus[i, :] = self.get_vec(yout.flatten() - yref.flatten())
            self.ZPsi[i, :] = self.get_vec(np.concatenate((yout.flatten()-yref.flatten(), a.flatten())))

        yref = self.get_random_traj(Sigma)
        a = self.get_u(K, yref)
        yout, _ = self.env.one_step(a.flatten())
        self.ZPhi_plus[self.tau_s, :] = self.get_vec(yout.flatten() - yref.flatten())
        
        

    def get_l_loss(self, y1: Array, y2: Array) -> float:
        """
        """
        dy = torch.from_numpy(y1 - y2).float().to(self.device)
        return torch.dot(dy, dy)

    def get_vec(self, x: Array) -> Array:
        """
        """
        x_tensor = torch.from_numpy(x.flatten()).float().to(self.device)
        return torch.outer(x_tensor, x_tensor).flatten()

    def run_mult_rounds(self, K: Array2D) -> None:
        """
        """
        for i in range(self.Tv):
            self.cur_it += 1

            yref = self.get_traj()
            a = self.get_u(K, yref)
            yout, _ = self.env.one_step(a.flatten())

            self.c[i] = self.get_l_loss(yref.flatten(), yout.flatten())
            self.Phi[i, :] = self.get_vec(yout.flatten() - yref.flatten())
            self.W[i, :] = self.w.flatten()

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
    
        self.cur_it += 1

        yref = self.get_traj()
        a = self.get_u(K, yref)
        yout, _ = self.env.one_step(a.flatten())

        self.Phi[self.Tv, :] = self.get_vec(yout.flatten() - yref.flatten())
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

    @staticmethod
    def get_h_hat(Phi1, Phi2, W, c) -> Array:
        """
        """
        Phi_inv = torch.linalg.pinv(Phi1.t() @ (Phi1 - Phi2 + W))
        return Phi_inv.t() @ Phi1.t() @ c

    def get_G(self, Psi, Phi, c, W, h_hat):
        """
        """
        # Psi_inv = torch.linalg.inv(Psi.t() @ Psi + self.E) @ Psi.t()
        Psi_inv = torch.linalg.pinv(Psi)
        _c = c + (Phi - W) @ h_hat
        vec_G = Psi_inv @ _c
        return vec_G.reshape(self.l * 2, self.l * 2)

    def update_policy(self, G):
        """
        """
        G22 = G[self.l:, self.l:]
        G12 = G[:self.l, self.l:]
        result_tensor = -torch.linalg.pinv(G22) @ G12.t()
        return result_tensor.cpu().numpy()

    def save_data(self, iteration: int,
                  **kwargs) -> None:
        """Save the data
        """
        file_name = str(iteration)
        path_data = os.path.join(self.path_data, file_name)
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)

    def learning(self) -> None:
        G = None
        if self.mode == 'v1':
            self.collect_data(self.K, self.Sigma_a)
        
        for i in range(self.S):
            self.run_mult_rounds(self.K)
            h_hat = self.get_h_hat(self.Phi[:-1, :], self.Phi[1:, :], self.W, self.c)

            if self.mode == 'v2':
                self.collect_data(self.K, self.Sigma_a)
            
            if G is None:
                G = self.get_G(self.ZPsi, self.ZPhi_plus[1:, :], self.Zc, self.ZW, h_hat)
            else:
                G += self.get_G(self.ZPsi, self.ZPhi_plus[1:, :], self.Zc, self.ZW, h_hat)

            self.K -= self.update_policy(G)
            
        
