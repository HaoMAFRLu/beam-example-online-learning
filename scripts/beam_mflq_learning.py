"""
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mflq_learning import MFLQ

random.seed(9527)
torch.manual_seed(9527)

def main(mode,
         T,
         xi,
         exp_name,
         _w,
         _sigma_a):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = MFLQ(mode=mode,
                    T=T,
                    xi=xi,
                    exp_name=exp_name,
                    _w=_w,
                    _sigma_a=_sigma_a,
                    is_vis=False)
    
    learning.learning()

if __name__ == '__main__':
    main(mode='v1',
         T=1000,
         xi=0.00001,
         exp_name='mflqv1',
         _w=0.3,
         _sigma_a=0.5)