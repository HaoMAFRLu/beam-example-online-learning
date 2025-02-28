"""Classes for the online optimizer
"""
import numpy as np
import time
import torch

from mytypes import Array, Array2D
import utils as fcs
from step_size import StepSize

class OnlineOptimizer():
    """The class for online quais-Newton method

    parameters:
    -----------
    mode: gradient descent method or newton method
    B: identified linear model
    """
    def __init__(self, mode: str, B: Array2D,
                 alpha: float, epsilon: float, 
                 eta: float, gamma: float,
                 rolling: int=50) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.B = self.move_to_device(B)
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.nr_iteration = 0
        self.rolling = rolling
        self.Lambda_list = []
        self.omega_list = []
        self.step_size = StepSize('constant', {'value0': self.eta})

    def ini_matrix(self, dim: int) -> None:
        """Initialize the matrices
        """
        if self.mode == 'newton':
            self.A = self.move_to_device(np.zeros((dim, dim)))
            self.I = self.move_to_device(np.eye(dim))
        elif self.mode == 'gradient':
            self.A = []

    def move_to_device(self, data: Array) -> torch.Tensor:
        """Move data to the device
        """ 
        return torch.from_numpy(data).to(self.device).float()

    def import_omega(self, data: torch.Tensor) -> None:
        self.omega = data.clone().view(-1, 1)

    def import_par_pi_par_omega(self, data: torch.Tensor) -> None:
        """Import par_pi_par_omega
        """
        self.par_pi_par_omega = data.clone()
    
    @staticmethod
    def get_L(B, par_pi_par_omega) -> torch.Tensor:
        """Get the L matrix
        """
        return torch.matmul(B, par_pi_par_omega)
    
    @staticmethod
    def get_Lambda(L: torch.Tensor, par_pi_par_omega: torch.Tensor, I: torch.Tensor,
                   alpha: float, epsilon: float) -> torch.Tensor:
        """Get the single pseudo Hessian matrix
        """
        return torch.matmul(L.t(), L) + alpha*torch.matmul(par_pi_par_omega.t(), par_pi_par_omega) + epsilon*I 

    def update_Lambda(self, B: torch.Tensor) -> None:
        """Update Lambda list
        """
        self.L = self.get_L(B, self.par_pi_par_omega)
        self.Lambda_list.append(self.get_Lambda(self.L, self.par_pi_par_omega, self.I, self.alpha, self.epsilon))
        if len(self.Lambda_list) > self.rolling:
            self.Lambda_list.pop(0)
        
    def update_A(self, B: torch.Tensor):
        """Update the pseudo Hessian matrix
        """
        self.update_Lambda(B)     
        self.A = sum(self.Lambda_list)/len(self.Lambda_list)

    @staticmethod
    def get_gradient(L: torch.Tensor, 
                     yref: torch.Tensor, yout: torch.Tensor) -> torch.Tensor:
        """
        """
        return torch.matmul(L.t(), yout - yref)

    def clear_A(self) -> None:
        self.A = self.A*0.0
        self.Lambda_list = []

    def update_model(self) -> torch.Tensor:
        l = self.yref.shape[0]
        return self.B[0:l, 0:l]

    def _optimize_newton(self) -> None:
        """Optimize the parameters using newton method
        """
        self._B = self.update_model()
        self.update_A(self._B)
        self.eta = self.step_size.get_eta(self.nr_iteration)

        self.gradient = self.get_gradient(self.L, self.yref, self.yout)        
        self.omega -= self.eta*torch.matmul(torch.linalg.inv(self.A), self.gradient)

    def _optimize_gradient(self) -> None:
        """Optimize the parameters using gradient descent method
        """
        self._B = self.update_model()
        self.L = self.get_L(self._B, self.par_pi_par_omega)
        self.gradient = self.get_gradient(self.L, self.yref, self.yout)
        self.eta = self.step_size.get_eta(self.nr_iteration)
        self.omega -= self.eta*self.gradient

    def save_latest_omega(self) -> None:
        """Save the latest well-trained parameters, when
        the distribution shift detected
        """
        self.omega_list.append(self.omega.clone())

    @staticmethod
    def rbf_kernel(x1: Array, x2: Array, gamma):
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff, diff))

    def get_kernel(self, y: Array, y_list: list) -> Array:
        """
        """
        k = []
        for y_ in y_list:
            k_ = self.rbf_kernel(y.flatten(), y_.flatten(), self.gamma)
            k.append(k_)
        return k 
    
    @staticmethod
    def get_omega(k, omega) -> torch.Tensor:
        """
        """
        return sum([x * y for x, y in zip(k, omega)])

    @staticmethod
    def normalize(lst):
        total = sum(lst)
        if total == 0:
            normalized_lst = [0 for x in lst]
        else:
            normalized_lst = [x / total for x in lst]
        return normalized_lst
    
    def initialize_omega(self, ydec: Array, yout_list: list) -> None:
        """Initialize the parameters using kernel method
        """
        k = self.get_kernel(ydec, yout_list)
        k = self.normalize(k)
        self.omega = self.get_omega(k, self.omega_list)

    def optimize(self, yref: Array, yout: Array) -> Array:
        """Do the online optimization
        """
        self.nr_iteration += 1

        self.yref = self.move_to_device(yref.reshape(-1, 1))
        self.yout = self.move_to_device(yout.reshape(-1, 1))

        if self.mode == 'gradient':
            self._optimize_gradient()
        elif self.mode == 'newton':
            self._optimize_newton()
