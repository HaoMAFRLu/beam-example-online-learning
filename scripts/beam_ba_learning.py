"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ba_learning import BA

random.seed(9527)
torch.manual_seed(9527)

def main(Ts,
         Ti,
         eta,
         sigma,
         exp_name):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = BA(Ts=Ts,
                  Ti=Ti,
                  eta=eta,
                  sigma=sigma,
                  exp_name=exp_name,
                  is_vis=False)
    
    learning.learning()

if __name__ == '__main__':
    main(Ts=100,
         Ti=20,
         eta=1.0,
         sigma=0.5,
         exp_name='ba')