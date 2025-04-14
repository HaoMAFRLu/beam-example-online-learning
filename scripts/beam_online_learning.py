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

def main(T: int,
         alpha: float,
         epsilon: float,
         eta: float):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    online_learning = OnlineLearning(mode='newton',
                                     exp_name='online_learning',
                                     alpha=alpha,
                                     epsilon=epsilon,
                                     eta=eta,
                                     is_vis=False)
    
    online_learning.online_learning(T)

if __name__ == '__main__':
    main(T=1000,
         alpha = 1.0,
         epsilon = 0.10,
         eta = 25.0)