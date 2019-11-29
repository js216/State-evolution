import sys
import json
import time
import hashlib
import datetime
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt

def plot(run_dir, options_fname, title="", ax=None):
    # import run options dict
    with open(run_dir+"/options/"+options_fname) as options_file:
        option_dict = json.load(options_file)

    # plot parameters
    scan_values  = np.linspace(**option_dict["scan_range"])
    scan_param   = option_dict["scan_param"]
    state_idx    = option_dict["state_idx"]
    fixed_params = option_dict["fixed_params"]

    # load results
    results_md5 = hashlib.md5(open(run_dir+"/options/"+options_fname,'rb').read()).hexdigest()
    with open(run_dir+"/results/"+results_md5+".txt") as f:
        eval_time, results = json.load(f)

    # plot results
    if ax is None:
        ax = plt.gca()
    ax.plot(scan_values, results, lw=2, color="black")
    units = {
        "DCi"     : "V/cm",
        "DCslope" : "V/cm/s",
        "ACi"     : "V/cm",
        "deltaT"  : "s",
        "ACw"     : "Hz",
    }
    ax.set_xlabel(scan_param+" ["+units[scan_param]+"]")
    ax.set_ylabel("$P_\mathrm{exit}$ from state "+str(state_idx))
    longtitle = ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) for (key, value) in fixed_params.items()])
    ax.set_title("\n".join(wrap(title+longtitle, 45)), fontdict={'fontsize':16})
    ax.grid()
    final_time = "eval time = "+str(datetime.timedelta(seconds=round(eval_time)))
    ax.text(1.01, .05, final_time, transform=ax.transAxes, rotation='vertical', fontdict={'fontsize':10})

    # save plot to file
    plt.savefig(run_dir+"/plots/"+results_md5+".png")

if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2], title="")
