"""The script is used to train the beam example to track
any given reference trajectories in an online manner. Mainly
follow this paper: https://arxiv.org/abs/2404.05318.
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning

random.seed(9527)
torch.manual_seed(9527)

def main(mode:str,
         T: int,
         exp_name: str,
         alpha: float,
         epsilon: float,
         eta: float,
         is_vis: bool,
         learn_mode: str):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    online_learning = OnlineLearning(mode=mode,
                                     exp_name=exp_name,
                                     alpha=alpha,
                                     epsilon=epsilon,
                                     eta=eta,
                                     is_vis=is_vis,
                                     learn_mode=learn_mode)
    
    online_learning.online_learning(T)

if __name__ == '__main__':
    main(mode='newton',
         T=1000,
         exp_name='newton',
         alpha=0.1,
         epsilon=1.0,
         eta=15.0,
         is_vis=False,
         learn_mode='s')