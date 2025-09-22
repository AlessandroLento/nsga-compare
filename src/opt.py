"""Main runner for NSGA-II vs NSGA-III experiments using pymoo.
Usage: import opt and call run_experiment(...) or run via main_test.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.utils import (
                    ensure_dir,
                    get_sampling, get_selection,
                    get_crossover, get_mutation
                    )

from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions


pop_default = 200
seed_default = 20
n_gen_default = 50


def build_algorithm(name, config, n_var, n_obj):

    #TODO: rendere anche get_reference_directions un parametro in config
    # config: dict with keys like population_size, sampling, crossover, mutation
    
    pop = config.get('population_size', pop_default)
    
    if name.lower() == 'nsga2':
        
        alg = NSGA2(pop_size=pop,
                    sampling=get_sampling(config.get("sampling"),None),
                    selection=get_selection(**config.get("selection",{'name':None})),
                    crossover=get_crossover(**config.get("crossover",{'name':None})),
                    mutation=get_mutation(**config.get("mutation",{'name':None})),
                    eliminate_duplicates=(config.get("eliminate_duplicates",True))
                    )
    
    elif name.lower() == 'nsga3':
        
        ref_dirs = None

        try:
            ref_dirs = get_reference_directions(config.get("method_reference_directions","uniform"), 
                                        n_obj, n_partitions=config.get('ref_partitions', 12)) 
        except Exception:
            ref_dirs = None

        alg = NSGA3(pop_size=pop,
                    ref_dirs=ref_dirs,
                    sampling=get_sampling(config.get("sampling"),None),
                    selection=get_selection(**config.get("selection",{'name':None})),
                    crossover=get_crossover(**config.get("crossover",{'name':None})),
                    mutation=get_mutation(**config.get("mutation",{'name':None})),
                    eliminate_duplicates=(config.get("eliminate_duplicates",True))
                    )
    
    else:
        raise ValueError(f"Unknown algorithm {name}")
    
    return alg



def run_single(problem_name, algorithm_name, config, out_dir):
    
    #TODO: rendere anche get_termination un parametro in config 

    
    ensure_dir(out_dir)
    problem = get_problem(problem_name)
    n_var = problem.n_var
    n_obj = problem.n_obj
    
    alg = build_algorithm(algorithm_name, config, n_var, n_obj)
    termination = get_termination("n_gen", config.get('n_gen', n_gen_default)) 
    
    res = minimize(problem,
                   alg,
                   termination,
                   seed=config.get('seed', seed_default),
                   save_history=False,
                   verbose=False)
                   
    # save results
    X = res.X    
    F = res.F
    G = res.G
    np.save(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_F.npy"), F)
    np.save(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_G.npy"), G)

    #metrics = save_metrics(F, problem, out_dir, problem_name, algorithm_name)

    
    return res
    
    

def run_experiments(problem_list, config_files, result_root='result'):
    
    for cfg_path in config_files:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
    
        for problem_name in problem_list:
                for alg_name in cfg.get('algorithms', ['nsga2','nsga3']):
                    
                    out_dir = os.path.join(result_root, os.path.basename(cfg_path).replace('.json',''), problem_name, alg_name)
                    print(f"Running {alg_name} on {problem_name} with config {cfg_path} -> {out_dir}")
                    run_single(problem_name, alg_name, cfg.get('algo_params', {}), out_dir)







if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run NSGA experiments')
    parser.add_argument('--problems', nargs='+', default=['zdt1','dtlz2'], help='problem names recognized by pymoo.get_problem')
    parser.add_argument('--configs', nargs='+', default=['test/payload/config1.json'], help='config json files')
    parser.add_argument('--result', default='result', help='result folder')
    args = parser.parse_args()
    run_experiments(args.problems, args.configs, args.result)
    analyse_results()