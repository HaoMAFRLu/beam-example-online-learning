"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from rac_learning import RAC

random.seed(9527)
torch.manual_seed(9527)

def main(Ts,
         Ti,
         exp_name):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = RAC(Ts=Ts,
                   Ti=Ti,
                   exp_name=exp_name,
                   is_vis=False)
    
    learning.learning()

if __name__ == '__main__':
    main(Ts=50,
         Ti=80,
         exp_name='rac30')