"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from linear_learning import LINEAR

random.seed(9527)
torch.manual_seed(9527)

def main(T: int,
         exp_name: str):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = LINEAR(T=T,
                  exp_name=exp_name,
                  is_vis=True)
    
    learning.learning()

if __name__ == '__main__':
    main(T=1000,
         exp_name='linear')