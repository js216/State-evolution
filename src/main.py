import time
import os, sys
import datetime
import numpy as np
import json, hashlib
from tqdm import tqdm
from mpi4py import MPI
from textwrap import wrap
from functools import reduce
import matplotlib.pyplot as plt

import TlF
import fields
from util import expm_arr

# default MPI communicator
COMM = MPI.COMM_WORLD

def run_scan(val_range, H_fname, fixed_params, time_params, state_idx,
             scan_param, scan_param2="none", batch_size=16384, **kwargs):
    # import Hamiltonian
    H_fn = TlF.load_Hamiltonian(H_fname)

    exit_probs = []
    for val1,val2 in tqdm(val_range):
        # define time grid and split into batches
        all_time   = fields.time_mesh(**fixed_params, **{scan_param: val1, scan_param2: val2}, **time_params)
        num_batches = 1 if batch_size >= len(all_time) else len(all_time)//batch_size
        t_batches  = np.array_split(all_time[:-1], num_batches)
        dt_batches = np.array_split(np.diff(all_time), num_batches)

        # calculate time-evolution operator
        U = np.eye(H_fn([[0,0,0,0,0,0]]).shape[-1])
        for t_arr, dt_arr in zip(t_batches, dt_batches):
            field_arr = fields.field(t_arr, **fixed_params, **{scan_param: val1, scan_param2: val2})
            dU = expm_arr(-1j * dt_arr[:,np.newaxis,np.newaxis] * H_fn(field_arr), s=time_params["s"])
            U = U @ reduce(np.matmul, dU)

        # evaluate transition probability
        _, P = np.linalg.eigh(H_fn([field_arr[-1]])[0])
        trans = np.abs(P @ U @ np.linalg.inv(P))**2
        exit_probs.append( 1 - trans[state_idx][state_idx] )

    return exit_probs


def process(result_fname, scan_range, scan_range2=None, **scan_params):
    if COMM.rank == 0:
        # flatten a 2D scan (if applicable)
        range1 = np.linspace(**scan_range)
        range2 = np.linspace(**scan_range2) if scan_range2 else [0]
        scan_space = np.dstack(np.meshgrid(range1, range2, indexing='ij')).reshape(-1, 2)

        # shuffle the scan space (to distribute workload more uniformly)
        permutation = np.random.permutation(scan_space.shape[0])
        scan_space = scan_space[permutation]

        # for runtime analysis
        start_time = time.time()
        num_timesteps = estimate_runtime(scan_space, **option_dict)
        print(num_timesteps)

        # split job into specified number of chunks
        scan_chunks = np.array_split(scan_space, COMM.size)
    else:
        scan_chunks = None

    # scatter jobs across cores
    in_chunk   = COMM.scatter(scan_chunks, root=0)
    out_chunk  = run_scan(in_chunk, **scan_params)
    exit_probs = MPI.COMM_WORLD.gather(out_chunk, root=0)

    # collect the results together
    if COMM.rank == 0:
        # unshuffle
        shuffled_results = np.array(sum(exit_probs,[])) #flatten
        unshuffled_results = np.zeros(shuffled_results.shape)
        unshuffled_results[permutation] = shuffled_results

        # write to file
        with open(result_fname, "w") as f:
            json.dump([
                num_timesteps,
                time.time()-start_time,
                list(unshuffled_results)
            ], f)


def plot(run_dir, options_fname, title="", vmin=None, vmax=None):
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

    # define units
    units = {
        "DCi"     : "V/cm",
        "DCslope" : "V/cm/s",
        "ACi"     : "V/cm",
        "deltaT"  : "s",
        "ACw"     : "Hz",
    }

    # import run options dict
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # load results
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    with open(run_dir+"/results/"+options_fname[:-5]+"-"+results_md5+".txt") as f:
        num_timesteps, eval_time, results = json.load(f)

    # 2D scan
    if "scan_range2" in option_dict:
        range1 = np.linspace(**option_dict["scan_range"])
        range2 = np.linspace(**option_dict["scan_range2"])
        X, Y = np.meshgrid(range2, range1)
        Z    = np.reshape(results, X.shape, order='F')
        plt.pcolormesh(Y, X, Z, cmap="nipy_spectral", vmin=vmin, vmax=vmax)
        plt.xlabel(option_dict["scan_param"]+" ["+units[option_dict["scan_param"]]+"]")
        plt.ylabel(option_dict["scan_param2"]+" ["+units[option_dict["scan_param2"]]+"]")
        plt.colorbar()

    # 1D scan
    else:
       plt.plot(np.linspace(**option_dict["scan_range"]), results, lw=2, color="black")
       plt.xlabel(option_dict["scan_param"]+" ["+units[option_dict["scan_param"]]+"]")
       plt.ylabel("$P_\mathrm{exit}$ from state "+str(option_dict["state_idx"]))

    # plot labels
    longtitle = title + option_dict["H_fname"].split("/")[-1] + ", "
    longtitle += ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) \
            for (key, value) in option_dict["fixed_params"].items()])
    longtitle += ", " + ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) \
            for (key, value) in option_dict["time_params"].items()])
    plt.title("\n".join(wrap(longtitle, 45)), fontdict={'fontsize':16})
    final_time = "eval time = "+str(datetime.timedelta(seconds=round(eval_time)))
    final_time += " @ " + '%.2e' % num_timesteps + " steps"
    plt.text(1.01, .05, final_time, transform=plt.gca().transAxes, rotation='vertical', fontdict={'fontsize':10})

    # save plot to file
    plt.grid()
    plt.tight_layout()
    plt.savefig(run_dir+"/plots/"+options_fname[:-5]+"-"+results_md5+".png")
    plt.close()


def estimate_runtime(val_range, fixed_params, time_params, scan_param, scan_param2="none", **kwargs):
    num_timesteps = 0
    for val1,val2 in val_range:
        time_grid = fields.time_mesh(**fixed_params, **{scan_param: val1, scan_param2: val2}, **time_params)
        num_timesteps += len(time_grid)
    return num_timesteps


if __name__ == '__main__':
    # decode script arguments
    run_dir       = sys.argv[1]
    options_fname = sys.argv[2]

    # import run options
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # check file hasn't been processed yet
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    results_fname = run_dir+"/results/"+options_fname[:-5]+"-"+results_md5+".txt"
    done = os.path.isfile(results_fname)

    # process the run
    if not done:
        process(results_fname, **option_dict)
    else:
        print("File already processed: " + options_fname)

    # plot results
    if COMM.rank == 0:
        plot(run_dir, options_fname, title="")
