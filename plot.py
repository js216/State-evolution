import sys
import json
import hashlib
import datetime
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt

def plot(options_file_fname, results_dir="results", plots_dir="plots", title=""):
    # import run options dict
    with open(options_file_fname) as options_file:
        option_dict = json.load(options_file)

    # plot parameters
    scan_values  = np.linspace(**option_dict["scan_range"])
    scan_param   = option_dict["scan_param"]
    state_idx    = option_dict["state_idx"]
    fixed_params = option_dict["fixed_params"]

    # load results
    results_md5 = hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest()
    results = np.loadtxt(results_dir+"/"+results_md5+".txt")

    # plot results
    plt.plot(scan_values, results, lw=2, color="black")
    units = {
        "DCi"     : "V/cm",
        "DCslope" : "V/cm/s",
        "ACi"     : "V/cm",
        "deltaT"  : "s",
        "ACw"     : "Hz",
    }
    plt.xlabel(scan_param+" ["+units[scan_param]+"]")
    plt.ylabel("$P_\mathrm{exit}$ from state "+str(state_idx))
    longtitle = ',  '.join(['%s\xa0=\xa0%.2g' % (key, value) for (key, value) in fixed_params.items()])
    plt.title("\n".join(wrap(title+longtitle, 45)), fontdict={'fontsize':16})
    plt.grid()
    plt.savefig(plots_dir+"/"+results_md5+".png")

if __name__ == '__main__':
    plot(options_file_fname=sys.argv[1])
