import sys
import time
import numpy as np
import json, hashlib
from tqdm import tqdm
from scipy.linalg import expm

import TlF
import fields

def run_scan(H_fname, scan_param, scan_range, fixed_params, time_params, state_idx=19, ax=None, title=""):
    start_time = time.time()

    # import the Hamiltonian matrix elements
    H_fn = TlF.load_Hamiltonian(H_fname)

    exit_probs = []
    for val in tqdm(np.linspace(**scan_range)):
        # define time grid, field, and Hamiltonian
        time_grid = fields.time_mesh(**fixed_params, **{scan_param: val}, **time_params)
        E_t = lambda t: fields.field(t, **fixed_params, **{scan_param: val})
        H  = lambda t: H_fn(E_t(t))[0]

        # calculate time-evolution operator
        dt = np.diff(time_grid)
        U = expm(-1j*dt[0]*H(0))
        for t,dt in zip(time_grid[1:-1], dt[1:]):
            U = expm(-1j*dt*H(t)) @ U

        # evaluate transition probability
        _, P = np.linalg.eigh(H(time_grid[-1]))
        trans = np.abs(P @ U @ np.linalg.inv(P))**2
        exit_probs.append(1 - trans[state_idx][state_idx])
    
    return time.time()-start_time, exit_probs

if __name__ == '__main__':
    # decode script parameters
    run_dir = sys.argv[1]
    options_fname = sys.argv[2]

    # import run options dict
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # run scan
    eval_time, results = run_scan(**option_dict)

    # write results to file
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    with open(run_dir+"/results/"+results_md5+".txt", "w") as f:
        json.dump([eval_time, results], f)
