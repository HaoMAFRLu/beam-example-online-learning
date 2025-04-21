"""
"""
import os, sys
import torch
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ac_learning import AC

random.seed(9527)
torch.manual_seed(9527)

def main(T: int,
         H: int,
         exp_name: str,
         eta: float,
         kappa: float,
         gamma: float,
         is_vis: bool,
         learn_mode: str):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = AC(T=T,
                  H=H,
                  exp_name=exp_name,
                  eta=eta,
                  kappa=kappa,
                  gamma=gamma,
                  is_vis=is_vis,
                  learn_mode=learn_mode)
    
    learning.learning()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="AC Learning")
    # parser.add_argument('--H', type=float, required=True, help="H")
    # args = parser.parse_args()

    main(T=1000,
         H=100,
         exp_name='ac',
         eta=50.0,
         kappa=5.0,
         gamma=0.05,
         is_vis=False,
         learn_mode='m')