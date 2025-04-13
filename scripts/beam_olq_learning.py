"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from olq_learning import OLQ

random.seed(9527)
torch.manual_seed(9527)

def main(T: int,
         eta: float,
         exp_name: str,
         nu: float):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = OLQ(exp_name=exp_name,
                   eta=eta,
                   nu=nu,
                   is_vis=True)
    
    learning.learning()

if __name__ == '__main__':
    main(T=500,
         eta=0.1,
         exp_name='olq',
         nu=200)