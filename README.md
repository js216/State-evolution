<p align="center">
  <img width="300" src="https://raw.githubusercontent.com/ograsdijk/CeNTREX/master/CeNTREX%20logo.png">
</p>

# State evolution scanning

This is a collection of Python scripts indented to study level transitions in
TlF ground state as a function of various parameters (e.g. electric fields).
More precisely, given a Hamiltonian `H` and `field(t, ...)`, it calculates the
probability the molecule exits a given final state.

The calculation proceeds in two steps. First, the time-evolution operator `U` is
calculated as the time-ordered exponential

    U = T exp{ ∫ H(t) dt },

formally solving the Schroedinger equation. Then, we find the transition
matrix elements

    T_ij = | ⟨i|U|j⟩ |^2,

where ⟨i| is the i-th eigenstate in the basis of final fields, and |j⟩ is the
j-th state in the final basis. The "exit probability" from state `i`, finally,
is defined as 1 − `T_ii`.

### Code structure

The Hamiltonian is defined in `TlF.py`, and is given almost entirely without
comments. For a thorough explanation of how the matrix elements are calculated,
peruse [this Jupyter notebook](https://github.com/js216/TlF-ground-state-Hamiltonian).
Note that the present code collection only requires one function to be defined,
`load_Hamiltonian(fname)`, which has to return a Hamiltonian `H(fields)` for a
list of fields `[[Ex, Ey, Ez, Bx, By, Bz], ...]`.

`main.py` is the script to be used to run a scan over parameters. Note
the final `if __name__ == '__main__'` statement, requiring two arguments to the
script: the "run directory", and the filename of the "option file" (to be
explained in the next section). The scan results are written in the appropriate
place in the `run_dir`, and the plots into `plots`.

### Basic usage

1. Create a folder ("`run_dir`") with the subdirectories `options`, `plots`, and
   `results`. Optionally, create a `slurm` folder for job-submission scripts
   when running the program on a cluster. An example script is provided, but see
   [here](https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/) for
   details about job scheduling @ Yale.

2. In the `options` subdirectory, place a file with the extension `.json`,
   defining the scan parameters. Consult the provided example files and the next
   section to determine what sections to define.

3. Run the `main.py` script, giving it the run directory, and the `.json` file
   as arguments. For example:

       python3 src/main.py runs/example/ example.json

   The program prints the number of time steps to be used with the chosen time
   mesh, and the progress of the calculation (using `tqdm`).

### Options file

The so-called options file has to be formatted as a JSON file, and be a dict
containing the following keys:

- `H_fname`, giving the filename of matrix elements of the chosen Hamiltonian
  in the `numpy` format. Four such files are already provided in the
  directory [matrices](https://github.com/js216/State-evolution/tree/master/matrices);
  for details about these files, look at `load_Hamiltonian()`, defined in
  `TlF.py`

- `field_str` a list of six strings, each defines the Python code to run to
  generate the corresponding field component as a function of time `t` and
  fixed/scannable parameters (as defined in the next few fields)

- `scan_param` and `scan_range`, giving the parameter that is to be scanned
  over

- optionally, if executing a 2D scan, `scan_param2` and `scan_range2` give
  the other scanned parameter

- `fixed_params`, giving the names and values of the fixed parameters that
  are to be passed to the field functions, as well as displayed in the plots

- optionally, `pickled_fnames` gives a dict of filename values associated with
  variable name keys; before the scan begins, the filenames are opened as `f`
  and the result of `pickle.load(f)` assigned to the given variable name. These
  variables are accessible to the `field_str` functions, as well as those in
  `time_params`, just like any other parameters.

- `units`, defining units for all parameters (used for labelling plots)

- `time_params` are used by `time_mesh()` together with all the field
  parameters to determine how fine a time mesh to make

- `state_idx` determines which state index (in the enumeration of eigenstates
  of `H(t=field(t_final))`) is used for calculating the exit probabilities

- `s` determines the level of approximation to be used for calculating matrix
  exponentials


### Time mesh details

The time mesh is defined by the four parameters inside the `time_params` dict:

- `t_final`: time evolution takes place between `t=0` and `t=t_final`
- `num_segm`: number of equal-size segments the entire evolution time from 0 to
  `t_final` is divided into (allowing each segment to have a different timestep
  density)
- `segm_pts`: number of points per segment, where the time endpoints of the
  segment are given as `T0` and `T1`
- `batch_size`: time evolution (i.e., field calculation, exponentiation of
  `-i*dt*H, matrix products) is done in batches of given size; size shouldn't
  affect the result, only the amount of memory used

If batch size is anything but 1, the special `expm_arr` function (defined in
`util.py`) is used to exponentiate the entire batch at once. That is useful for
small Hamiltonians on systems with a large parallel processing capability (i.e.,
many cores).

If batch size is not set to 1, then the `s` parameter (giving the number of
squarings in the expm algorithm) also has to be defined in the options file.
Whereas `scipy.linalg.expm` estimates `s` automatically, `expm_arr` does not.
Since all matrices are to be exponentiated in parallel, they all have to be
divided, scaled, and squared the same number of times, and that is given by the
`s` parameter.

### Parallel evaluation

The most time-consuming part of the calculation, matrix exponentiation, is
already parallelized by
[scipy.linalg.expm](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.linalg.expm.html)
in a multi-threaded way. Nonetheless, each scan with a decent number of steps
will take hours on a modern machine. Thus, I recommend defining multiple option
files within a single `run_dir`, and running each scan on a separated node in a
cluster. Moreover, the main script uses all available MPI ranks, so calling it
with `mpirun -n ⟨n⟩` will speed up the scan `n` times. Refer to the [YCRC
instructions](https://docs.ycrc.yale.edu/clusters-at-yale/) to learn how to
access the cluster at Yale.
