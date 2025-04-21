"""
"""
import os, sys
import torch
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ba_learning import BA

random.seed(9527)
torch.manual_seed(9527)

def main(Ts,
         Ti,
         eta,
         sigma,
         exp_name,
         is_vis,
         learn_mode):
    """For the specific meaning of the parameters 
    please refer to the paper.
    """ 
    learning = BA(Ts=Ts,
                  Ti=Ti,
                  eta=eta,
                  sigma=sigma,
                  exp_name=exp_name,
                  is_vis=is_vis,
                  learn_mode=learn_mode)
    
    learning.learning()

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="AC Learning")
    # parser.add_argument('--eta', type=float, required=True, help="eta")
    # parser.add_argument('--Ti', type=int, required=True, help="Ti")
    # args = parser.parse_args()

    main(Ts=1000,
         Ti=5,
         eta=1000.0,
         sigma=5.0,
         exp_name='ba',
         is_vis=False,
         learn_mode='m')