import glob
import os
import subprocess


from test.problem.problems import PROBLEMS
from src.opt import run_experiments
from src.utils import build_metrics, analyse_results





if __name__ == '__main__':
    
    cfgs = sorted(glob.glob('test/payload/*.json'))
       
    run_experiments(PROBLEMS, cfgs, result_root='result')
    
    build_metrics(PROBLEMS, cfgs, result_root='result') 

    analyse_results(result_root='result')
