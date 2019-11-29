import os
import sys
import json
import glob
import time
import hashlib
import datetime
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt

# general plot format
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

def plot(run_dir, options_fname, title="", ax=None):
    # import run options dict
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # load results
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    with open(run_dir+"/results/"+results_md5+".txt") as f:
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
    plt.savefig(run_dir+"/plots/"+results_md5+".png")
    plt.close()

if __name__ == '__main__':
    # decode script parameters
    run_dir = sys.argv[1]
    options_fnames = [sys.argv[2]] if len(sys.argv)>2 else glob.glob(run_dir+"/options/*.json")

    for path in options_fnames:
        plot(run_dir, os.path.basename(path), title="")
