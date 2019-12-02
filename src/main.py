import time
import os, sys
import datetime
import numpy as np
import json, hashlib
from tqdm import tqdm
from mpi4py import MPI
from textwrap import wrap
from scipy.linalg import expm
import matplotlib.pyplot as plt

import TlF
import fields

# default MPI communicator
COMM = MPI.COMM_WORLD

def run_scan(val_range, H_fname, scan_param, fixed_params, time_params, state_idx, **kwargs):
    # import Hamiltonian
    H_fn = TlF.load_Hamiltonian(H_fname)

    exit_probs = []
    for val in val_range:
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
        exit_probs.append( 1 - trans[state_idx][state_idx] )

    return exit_probs


def process(scan_range, **scan_params):
    if COMM.rank == 0:
        # for runtime analysis
        start_time = time.time()
        num_timesteps = estimate_runtime(np.linspace(**scan_range), **option_dict)

        # split job into specified number of chunks
        scan_chunks = np.split(np.linspace(**scan_range), COMM.size)
    else:
        scan_chunks = None

    # scatter jobs across cores
    in_chunk   = COMM.scatter(scan_chunks, root=0)
    out_chunk  = run_scan(in_chunk, **scan_params)
    exit_probs = MPI.COMM_WORLD.gather(out_chunk, root=0)

    # write results to file
    if COMM.rank == 0:
        results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
        with open(run_dir+"/results/"+options_fname[:-5]+"-"+results_md5+".txt", "w") as f:
            json.dump([num_timesteps, time.time()-start_time, list(np.ravel(exit_probs))], f)


def plot(run_dir, options_fname, title="", ax=None):
    # define plot format
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes.formatter', limits=(-5,5))  # tick labels go into scientific notation at 1e5, 1e-5

    # import run options dict
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # load results
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    with open(run_dir+"/results/"+options_fname[:-5]+"-"+results_md5+".txt") as f:
        num_timesteps, eval_time, results = json.load(f)

    # plot results
    if ax is None:
        ax = plt.gca()
    ax.plot(np.linspace(**option_dict["scan_range"]), results, lw=2, color="black")
    units = {
        "DCi"     : "V/cm",
        "DCslope" : "V/cm/s",
        "ACi"     : "V/cm",
        "deltaT"  : "s",
        "ACw"     : "Hz",
    }

    # plot labels
    ax.set_xlabel(option_dict["scan_param"]+" ["+units[option_dict["scan_param"]]+"]")
    ax.set_ylabel("$P_\mathrm{exit}$ from state "+str(option_dict["state_idx"]))
    longtitle = title + option_dict["H_fname"].split("/")[-1] + ", "
    longtitle += ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) \
            for (key, value) in option_dict["fixed_params"].items()])
    longtitle += ", " + ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) \
            for (key, value) in option_dict["time_params"].items()])
    ax.set_title("\n".join(wrap(longtitle, 45)), fontdict={'fontsize':16})
    final_time = "eval time = "+str(datetime.timedelta(seconds=round(eval_time)))
    final_time += " @ " + '%.2e' % num_timesteps + " steps"
    ax.text(1.01, .05, final_time, transform=ax.transAxes, rotation='vertical', fontdict={'fontsize':10})

    # save plot to file
    ax.grid()
    plt.tight_layout()
    plt.savefig(run_dir+"/plots/"+options_fname[:-5]+"-"+results_md5+".png")
    plt.close()


def estimate_runtime(val_range, scan_param, fixed_params, time_params, **kwargs):
    num_timesteps = 0
    for val in val_range:
        time_grid = fields.time_mesh(**fixed_params, **{scan_param: val}, **time_params)
        num_timesteps += len(time_grid)
    return num_timesteps


if __name__ == '__main__':
    # decode script arguments
    run_dir       = sys.argv[1]
    options_fname = sys.argv[2]

    # import run options
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # process the run
    process(**option_dict)

    # plot results
    if COMM.rank == 0:
        plot(run_dir, options_fname, title="", ax=None)
