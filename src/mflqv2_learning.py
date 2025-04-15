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
        self.E = np.eye(self.l)

        self.Sigma_a = np.eye(self.l) * self._sigma_a

        self.Phi = torch.zeros((self.Tv, self.l ** 2), dtype=torch.float32).to(self.device)
        self.c = torch.zeros((self.Tv, 1), dtype=torch.float32).to(self.device)
        self.set_cvxpy_1()
        self.set_cvxpy_2()

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
            self.Phi[i, :] = self.get_vec(yref.flatten())
    
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
    
    def get_G(self, Phi, c):
        """
        """
        # Psi_inv = torch.linalg.inv(Psi.t() @ Psi + self.E) @ Psi.t()
        Phi_inv = torch.linalg.pinv(Phi)
        vec_G = Phi_inv @ c
        return vec_G.reshape(self.l, self.l)

    def set_cvxpy_1(self):
        self.G_tilde = cp.Variable((self.l, self.l))
        self.G_param = cp.Parameter((self.l, self.l), name="G")
        # self.objective = cp.Minimize(cp.norm((self.B@self.K_tilde - self.E).T@(self.B@self.K_tilde - self.E) - self.G_param, 'fro'))
        objective = cp.Minimize(cp.norm(self.G_tilde.T@self.G_tilde - self.G_param, 'fro'))
        constraints = []
        self.prob1 = cp.Problem(objective, constraints)

    def set_cvxpy_2(self):
        self.K_tilde = cp.Variable((self.l, self.l))
        self.GT = cp.Parameter((self.l, self.l), name="GT")
        # self.objective = cp.Minimize(cp.norm((self.B@self.K_tilde - self.E).T@(self.B@self.K_tilde - self.E) - self.G_param, 'fro'))
        objective = cp.Minimize(cp.norm(self.B@self.K_tilde - self.E - self.GT, 'fro'))
        constraints = []
        self.prob2 = cp.Problem(objective, constraints)

    def solve1(self, G):
        self.G_param.value = G
        # self.prob.solve(solver=cp.CVXOPT, options={'abstol': 1e-2, 'reltol': 1e-1, 'feastol': 1e-1})
        # self.prob.solve(solver=cp.ECOS, tol=1e-2, feastol=1e-2, abstol=1e-2, reltol=1e-2)
        self.prob1.solve(solver=cp.SCS, eps=1e-2, max_iters=200, verbose=False)
        return self.G_tilde.value

    def update_policy(self, GT):
        """
        """
        self.GT.value = GT
        # self.prob.solve(solver=cp.CVXOPT, options={'abstol': 1e-2, 'reltol': 1e-1, 'feastol': 1e-1})
        # self.prob.solve(solver=cp.ECOS, tol=1e-2, feastol=1e-2, abstol=1e-2, reltol=1e-2)
        self.prob2.solve(solver=cp.SCS, eps=1e-2, max_iters=200, verbose=False)
        return self.K_tilde.value
    
    def save_data(self, iteration: int,
                  **kwargs) -> None:
        """Save the data
        """
        file_name = str(iteration)
        path_data = os.path.join(self.path_data, file_name)
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)

    def learning(self) -> None:

        for i in range(self.S):
            self.run_mult_rounds(self.K)
            G = self.get_G(self.Phi, self.c)
            GT = np.linalg.cholesky(G.cpu().numpy() + self.E)
            # GT = self.solve1(G.cpu().numpy())
            self.K = self.update_policy(GT.T)
            
        
