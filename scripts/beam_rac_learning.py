"""
"""
import os, sys
import torch
import random
import argparse

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

    parser = argparse.ArgumentParser(description="AC Learning")
    parser.add_argument('--Ti', type=int, required=True, help="Ti")
    args = parser.parse_args()

    main(Ts=500,
         Ti=args.Ti,
         exp_name='rac')