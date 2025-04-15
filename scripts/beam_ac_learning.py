"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ac_learning import AC

random.seed(9527)
torch.manual_seed(9527)

def main(T: int,
         H: int,
         exp_name: str,
         eta: float,
         kappa: float,
         gamma: float):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = AC(T=T,
                  H=H,
                  exp_name=exp_name,
                  eta=eta,
                  kappa=kappa,
                  gamma=gamma,
                  is_vis=True)
    
    learning.learning()

if __name__ == '__main__':
    main(T=1000,
         H=100,
         exp_name='ac',
         eta=100000.0,
         kappa=5.0,
         gamma=0.05)