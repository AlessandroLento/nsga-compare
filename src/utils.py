import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import json

from pymoo.problems import get_problem


from pymoo.operators.sampling.rnd import BinaryRandomSampling, FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.mutation.bitflip import BitflipMutation             
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random

from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from pymoo.util.ref_dirs import get_reference_directions


supported_metrics_gds=['GD','GDPlus','IGD','IGDPlus']
standard_methods_directions= ['uniform','das_dennis','incremental','energy']


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------- OPERATORI  (da configs)----------


def get_sampling(name,other_params=None):
    
    if name is None:
        return FloatRandomSampling()

    
    if name == "real_random":
        return FloatRandomSampling()
    elif name== "binary_random":
        return BinaryRandomSampling()
    elif name == "lhs":
        return LHS()

    
    raise ValueError(f"Sampling {name} non supportato.")




def get_selection(name, func_comp=comp_by_cv_then_random, pressure=2):
    
    if name is None:
        return TournamentSelection(pressure=pressure,func_comp=func_comp)

    
    if name == "tournament":
        return TournamentSelection(pressure=pressure,func_comp=func_comp)
        
    elif name == "random":
        return RandomSelection()

        
    raise ValueError(f"Selection {name} non supportato.")




def get_mutation(name, prob_poly=1.0, eta=20, repair=RoundingRepair,
                    prob_bitflip=0.5, prob_var_bitflip=0.3):
    
    if name is None:
        return PM(eta=eta)

    
    if name == "real_pm":
        return PM(eta=eta, prob=prob_poly)
    elif name == "discrete_pm":
        return PM(eta=eta, prob=prob_poly, repair=repair)
    elif name == "bitflip":
        return BitflipMutation(prob=prob_bitflip, prob_var=prob_var_bitflip)
        
    raise ValueError(f"Mutation {name} non supportato.")



def get_crossover(name, prob=1.0, prob_var=1.0, eta=30, repair=RoundingRepair,
                    k_point_crossover=4, prob_exp=0.9):
    
    if name is None:
        return SBX(prob=prob, eta=eta)

    
    
    if name == "real_sbx":
       return SBX(prob=prob, eta=eta, prob_var=prob_var)
    elif name == "discrete_sbx":
       return SBX(prob=prob, eta=eta, prob_var=prob_var, repair=repair, vtype=float)
       
    elif name == "one_point":
        return SinglePointCrossover(prob=prob)
    elif name == "two_point":
        return TwoPointCrossover(prob=prob)
    elif name == "multi_point":
        return PointCrossover(prob=prob, n_points= k_point_crossover)
        
    elif name == "exponential":
        return ExponentialCrossover(prob=prob, prob_exp=prob_exp)
        
    elif name == "uniform":
        return UniformCrossover(prob=prob)    
    elif name == "half_uniform":
        return HalfUniformCrossover(prob=prob) 

       
       
    raise ValueError(f"Crossover {name} non supportato.")


#TODO: def wrapper_get_reference_directions(n_obj, type='uniform', n_partitions=12, scaling=1.0):

        
    




# ---------- GRAFICI ----------


def scatter_plot(F, n_obj, problem_name, algorithm_name,out_dir):
    
        fig = plt.figure(figsize=(6,4))
        
        if n_obj == 2:
            plt.scatter(F[:,0], F[:,1], s=8)
            plt.xlabel('f1'); plt.ylabel('f2')
        
        elif n_obj == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(F[:,0], F[:,1], F[:,2], s=8)
            ax.set_xlabel('f1'); ax.set_ylabel('f2'); ax.set_zlabel('f3')
        
        plt.title(f"{problem_name} - {algorithm_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_front.png"))
        plt.close(fig)
        
        
def plot_front(F, problem_name, algorithm_name,out_dir, title="", true_pf=None):
    """Plot dei fronti: 2D o 3D."""
    n_obj = F.shape[1]
    fig = plt.figure(figsize=(6, 4))

    if n_obj == 2:
        plt.scatter(F[:, 0], F[:, 1], s=8, label="Approx. PF")
        if true_pf is not None:
            plt.scatter(true_pf[:, 0], true_pf[:, 1], s=8, c="r", alpha=0.5, label="True PF")
        plt.xlabel("f1"); plt.ylabel("f2")
        plt.legend()
    elif n_obj == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=8, label="Approx. PF")
        if true_pf is not None:
            ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=8, c="r", alpha=0.5, label="True PF")
        ax.set_xlabel("f1"); ax.set_ylabel("f2"); ax.set_zlabel("f3")
    else:
        plt.text(0.5, 0.5, f"{n_obj} objectives: no plot", ha="center")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_front2.png"))
    plt.close(fig)

        
        
def plot_front_pymoo(F, problem_name, algorithm_name,out_dir, title="", true_pf=None):

    #DEPRECATED
    #TODO:CORREGGERE
    
    """Plot dei fronti: 2D o 3D."""
    n_obj = F.shape[1]
    fig = plt.figure(figsize=(6, 4))

    if n_obj <=3:
        plot = Scatter()
        plot.add(F, s=30, facecolors='none', edgecolors='b')
        if true_pf is not None:

            plot.add(true_pf, plot_type="line", color="red", linewidth=2)
    
    else:

        plot = Scatter()
        plot.add(F, s=10)
        
        if true_pf is not None:

            plot.add(true_pf, s=30, color="red")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_front3.png"))
    plt.close(fig)





# ---------- METRICHE ----------


def compute_hv(F, ref_point=None, modifier_hv=1.1):

    """Calcola l'ipervolume."""
    if ref_point is None:
        ref_point = np.max(F, axis=0) * modifier_hv
    hv = HV(ref_point=ref_point)
    return hv(F)


def compute_gds(F, pf, type='IGD'):

    """Calcola varie GD rispetto al fronte di Pareto dato (pf)."""
    if type =='IGD':
        gdx = IGD(pf)
        return gdx(F)
    elif type =='IGDPlus':
        gdx = IGDPlus(pf)
        return gdx(F)
    elif type =='GD':
        gdx = GD(pf)
        return gdx(F)
    elif type =='GDPlus':
        gdx = GDPlus(pf)
        return gdx(F)


    raise ValueError(f"Metrica {type} non supportata.")



def load_results(config_name, problem_name, algorithm_name,
                result_root="result", type='F'):
    
    res_dir = os.path.join(config_name, problem_name, algorithm_name)
    
    value = np.load( os.path.join( result_root, res_dir,
    f"{problem_name}_{algorithm_name}_{type}.npy"))
    
    return (value)




def save_metrics(F, problem, out_dir, problem_name, algorithm_name):
 
    """Calcola e salva le metriche su file .npy"""
    
    metrics = {}
    
    # Ipervolume
    metrics["hv"] = compute_hv(F)

    # GDs se disponibile
    pf = problem.pareto_front() if hasattr(problem, "pareto_front") else None
    
    if pf is not None:
        metrics_gds = {type:compute_gds(F, pf, type) for type in supported_metrics_gds}
        
        metrics =  {**metrics, **metrics_gds}



    np.save(os.path.join(out_dir, f"{problem_name}_{algorithm_name}_metrics.npy"), metrics)

    return metrics



def load_metrics(result_root="result"):
    
    records = []
    # Ogni config ha una sottocartella
    for cfg_dir in sorted(glob.glob(os.path.join(result_root, "*"))):
        
        if not os.path.isdir(cfg_dir):
            continue
        config_name = os.path.basename(cfg_dir)
        
        # Dentro ogni config ci sono problemi
        for prob_dir in glob.glob(os.path.join(cfg_dir, "*")):
            
            problem_name = os.path.basename(prob_dir)
            
            for alg_dir in glob.glob(os.path.join(prob_dir, "*")):
                
                algorithm_name = os.path.basename(alg_dir)
                metrics_files = glob.glob(os.path.join(alg_dir, "*_metrics.npy"))
                
                for mf in metrics_files:
                    
                    metrics = np.load(mf, allow_pickle=True).item()
                    
                    records_base={
                        "config": config_name,
                        "problem": problem_name,
                        "algorithm": algorithm_name,
                        "hv": metrics.get("hv", None)
                    }
                    records_gds = {type:metrics.get(type, None) for type in supported_metrics_gds}
                    records_base = {**records_base, **records_gds}
                    
                    records.append(records_base)
                    

                    
    return pd.DataFrame(records)




def build_metrics(problem_list, config_files, result_root='result'):
    
    
    for cfg_path in config_files:
        
        cfg_alias = os.path.basename(cfg_path).replace('.json','')
        

        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        for problem_name in problem_list:
            
                for alg_name in cfg.get('algorithms', ['nsga2','nsga3']):
                                       

                    F = load_results(cfg_alias,problem_name,alg_name,
                                    result_root)
                                    
                    problem = get_problem(problem_name)

                                    
                    out_dir = os.path.join(result_root, cfg_alias, 
                                problem_name, alg_name)
                                    
                                                            
                    save_metrics(F, problem, out_dir,
                        problem_name, alg_name)
                        
                    print(f"Calculated metrics of {alg_name} on {problem_name} with config {cfg_path} -> {out_dir}")
                    
                    # scatter plot (only for 2 or 3 objectives)
                    
                    if problem.n_obj <= 3:
                        
                        pf = problem.pareto_front() if hasattr(problem, "pareto_front") else None
                        
                        scatter_plot(F, problem.n_obj,
                            problem_name,alg_name,out_dir)  
                            
                        plot_front(F, problem_name, alg_name, out_dir , 'Complete Fronts', pf)
                    
                        #plot_front_pymoo(F, problem_name, alg_name, out_dir , 'Complete Fronts', pf)






def analyse_results(result_root="result"):

        
    df = load_metrics(result_root) 
    
    
    if df.empty:
        print("Nessun risultato trovato in result/. Esegui prima main_test.py")
        return

    print("\n=== Riepilogo metriche ===")
    print(df)

    # Pivot per confronto leggibile
    pivot = df.pivot_table(index=["config", "problem"],
                           columns="algorithm",
                           values=["hv"]+supported_metrics_gds)
    print("\n=== Tabella comparativa (hv, gds) ===")
    print(pivot)

    out_dir = os.path.join(result_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    pivot.to_csv(os.path.join(out_dir, "metrics_pivot.csv"))
    print(f"\n Salvati i file in {out_dir}/metrics_summary.csv e metrics_pivot.csv")





# def run_metrics():
    
    # problems = glob.glob(os.path.join("test", "problem", "*.py"))
    # configs = glob.glob(os.path.join("test", "payload", "*.json"))

    # for prob in problems:
        # problem_name = os.path.splitext(os.path.basename(prob))[0]
        # for cfg in configs:
            # print(f"\n>>> Running {problem_name} with {cfg}")
            # subprocess.run(["python", "src/opt.py", "--problem", problem_name, "--config", cfg])




