import time
import pickle
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
from util import expm_arr, eval_num

# default MPI communicator
COMM = MPI.COMM_WORLD

def run_scan(val_range, H_fname, state_idx, scan_param, field_str, fixed_params,
        time_params, s=None, pickle_fnames=None, scan_param2="none",
        batch_size=16384, **kwargs):

    # import Hamiltonian
    H_fn = TlF.load_Hamiltonian(H_fname)

    # import pickled variables (if any)
    pickled_vars = {}
    if pickle_fnames:
        for key, fname in pickle_fnames.items():
            with open(fname, 'rb') as f:
                pickled_vars[key] = pickle.load(f)

    exit_probs = []
    for i,val1,val2 in tqdm(val_range):
        # check there is a parameter value
        if np.isnan(val1) or np.isnan(val2):
           continue

        # import parameters
        phys_params = {
            **{scan_param: val1, scan_param2: val2},
            **fixed_params, **time_params, **pickled_vars, **kwargs}

        # calculate time-evolution operator
        U = np.eye(H_fn([[0,0,0,0,0,0]]).shape[-1])
        t_batches, dt_batches = time_mesh(phys_params)
        for t, dt in zip(t_batches, dt_batches):
            H = H_fn(field(field_str, phys_params, t))
            dU = expm_arr(-1j * 2*np.pi * dt[:,np.newaxis,np.newaxis] * H, s)
            U =  reduce(np.matmul, dU[::-1])@ U

        # evaluate transition probability
        psi_i = eig_state(H_fn, field_str, phys_params, t_batches[0][0],   state_idx)
        psi_f = eig_state(H_fn, field_str, phys_params, t_batches[-1][-1], state_idx)
        exit_probs.append(1 - np.abs(psi_f.conj() @ U @ psi_i)**2)

    return np.hstack((val_range,exit_probs[:,np.newaxis]))


def process(result_fname, scan_range, scan_range2=None, **scan_params):
    """Distribute work and collect results.

    Arguments:
    result_fname:     text file for storing calculation results
    scan_range, etc.: parameters describing the scan (from options file)
    """
    if COMM.rank == 0:
       # flatten and enumerate a 2D scan (if applicable)
       range1 = np.linspace(**scan_range, dtype=float)
       range2 = np.linspace(**scan_range2) if scan_range2 else [0]
       scan_space = np.dstack(np.meshgrid(range1, range2, indexing='ij')).reshape(-1, 2)
       scan_space = np.hstack((np.arange(scan_space.shape[0])[:,np.newaxis], scan_space))

       # split job into specified number of equal-size chunks
       cs = scan_params["chunk_size"]
       pad_width   = ((0, cs-len(scan_space)%cs),(0, 0))
       scan_space  = np.pad(scan_space, pad_width, 'constant', constant_values=np.nan)
       scan_chunks = np.split(scan_space, scan_space.shape[0]//cs)

       # for keeping track of worker ranks
       workers = {i : [0, 0, np.empty((cs,4), dtype=np.float64)] for i in range(1,COMM.size)}

       # send work and receive results until scan finished
       with open(result_fname, "a") as f:
          while len(scan_chunks) > 0:
             for rank,(send_req,recv_req,data) in workers.items():
                # if chunk read by worker, send another one
                if send_req==0 or send_req.Test():
                   workers[rank][0] = MPI.Isend(scan_chunks.pop(), dest=rank)
                   workers[rank][1] = MPI.Irecv(workers[rank][2], source=rank)

                # if worker returned results, write them to file
                elif recv_req.Test():
                   np.savetxt(f, workers[rank][2])
             time.sleep(1)

    # for worker ranks
    else:
       data = numpy.empty((scan_params["chunk_size"],3), dtype=numpy.float64)
       while True:
          COMM.Recv(data, source=0)
          COMM.Send(run_scan(data, **option_dict), dest=0)

def time_mesh(phys_params):
    """Generate a time mesh given mesh parameters.

    Arguments:
    phys_params: dict with mesh-defining parameters

    The following elements are required in the dict:
    t_final:    final time of state evolution (str -> float)
    num_segm:   number of mesh segments (str -> int)
    segm_pts:   timesteps/time in interval [T0,T1] (str -> int)
    batch_size: maximum number of steps per batch

    Returns:
    t_batches:  time grid, batched
    dt_batches: time differences, batched
    """
    # split the time period into a number of segments
    segm = np.linspace(
            start = 0,
            stop  = eval_num(phys_params["t_final"],phys_params),
            num   = max(eval_num(phys_params["num_segm"],phys_params), 1)+1)

    # make a sub-mesh for each of the segments
    t = []
    for i in range(len(segm)-1):
        N_pts = eval_num(
              phys_params["segm_pts"],
              {**phys_params, 'T0':segm[i], 'T1':segm[i+1]})
        t.extend(np.linspace(segm[i], segm[i+1], int(N_pts)))

    # split into batches
    num_batches = max(1, len(t)//phys_params["batch_size"]-1)
    t_batches  = np.array_split(t[:-1],     num_batches)
    dt_batches = np.array_split(np.diff(t), num_batches)

    return t_batches, dt_batches


def field(field_str, phys_params, t_arr):
    """Return list of fields given a list of times."""
    return np.transpose([eval_num(x,{**phys_params,'t':t_arr}) for x in field_str])


def eig_state(H_fn, field_str, phys_params, t, state_idx):
    """Return eigenstate at a given time."""
    return np.linalg.eigh(H_fn(field(field_str,phys_params,np.array([t])))[0])[1][:,state_idx]


def import_options(run_dir, options_fname):
   # load options file
   with open(run_dir+"/options/"+options_fname) as options_file:
      option_dict = json.load(options_file)

   # check file hasn't been processed yet
   results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
   results_fname = run_dir+"/results/"+options_fname[:-5]+"-"+results_md5+".txt"
   done = os.path.isfile(results_fname)

   # return
   return done, result_fname, option_dict


def plot(run_dir, options_fname, vmin=None, vmax=None):
    """Plot calculation results.

    Arguments:
    run_dir:       directory containing information about a scan
    options_fname: filename within run_dir/options
    title:         optional extra text for plot title
    vmin:          optional vmin for 2D plots
    vmax:          optional vmax for 2D plots
    """
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
        results = json.load(f)

    # draw the plots
    if "scan_range2" in option_dict:
        range1 = np.linspace(**option_dict["scan_range"])
        range2 = np.linspace(**option_dict["scan_range2"])
        X, Y = np.meshgrid(range2, range1)
        Z    = np.reshape(results, X.shape, order='C')
        plt.pcolormesh(Y, X, Z, cmap="nipy_spectral", vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
       plt.plot(np.linspace(**option_dict["scan_range"]), results, lw=2, color="black")

    # plot labels
    longtitle = option_dict["H_fname"].split("/")[-1] + ", "
    longtitle += ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) \
            for (key, value) in option_dict["fixed_params"].items()])
    if option_dict.get("comment"):
        longtitle += str(option_dict.get("comment"))
    plt.title("\n".join(wrap(longtitle, 45)), fontdict={'fontsize':16}, pad=25)
    plt.text(1.00, 1.01, options_fname[:-5]+"-"+results_md5,
          transform=plt.gca().transAxes, fontdict={'fontsize':8}, ha="right")
    units = option_dict["units"]
    if "scan_range2" in option_dict:
       plt.xlabel(option_dict["scan_param"]+" ["+units[option_dict["scan_param"]]+"]")
       plt.ylabel(option_dict["scan_param2"]+" ["+units[option_dict["scan_param2"]]+"]")
    else:
       plt.xlabel(option_dict["scan_param"]+" ["+units[option_dict["scan_param"]]+"]")
       plt.ylabel("$P_\mathrm{exit}$ from state "+str(option_dict["state_idx"]))

    # save plot to file
    plt.grid()
    plt.tight_layout()
    plt.savefig(run_dir+"/plots/"+options_fname[:-5]+"-"+results_md5+".png")
    plt.close()


if __name__ == '__main__':
    # import run options
    done, result_fname, option_dict = import_options(
          run_dir       = sys.argv[1]
          options_fname = sys.argv[2])

    # process if necessary
    if not done:
        process(results_fname, **option_dict)

    # plot results
    plot(run_dir, options_fname)
