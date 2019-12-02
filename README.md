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

    T_ij = | ⟨j|U|i⟩ |^2

in the eigenbasis of `H` at the final fields. The "exit probability" from state
`i`, finally, is defined as 1 − `T_ii`.

### Code structure

The Hamiltonian is defined in `TlF.py`, and is given almost entirely without
comments. For a thorough explanation of how the matrix elements are calculated,
peruse [this Jupyter notebook](https://github.com/js216/TlF-ground-state-Hamiltonian).
Note that the present code collection only requires one function to be defined,
`load_Hamiltonian(fname)`, which has to return a Hamiltonian `H(fields)` for a
list of fields `[[Ex, Ey, Ez, Bx, By, Bz], ...]`.

The fields are defined in `fields.py` as a set of functions of time and
(scannable) parameters. The scan code only requires two to be defined:

- `field(t, **params)`: returns a `numpy` array of the fields at a particular
  point in time (see code for details)

- `time_mesh(**params)`: returns an appropriate time mesh on which to evaluate
  the state evolution (i.e., the time steps are the more densely packed when the
  fields and thus the Hamiltonian are expected to vary quickly with time)

Finally, `main.py` is the script to be used to run a scan over parameters. Note
the final `if __name__ == '__main__'` statement, requiring two arguments to the
script: the "run directory", and the filename of the "option file" (to be
explained in the next section). The scan results are written in the appropriate
place in the `run_dir`, and the plots into `plots`.

### Basic usage

1. Create a folder ("`run_dir`") with the subdirectories `options`, `plots`, and
   `results`.

2. In the `options` subdirectory, place a file with the extension `.json`,
   defining the scan parameters. Consulting the provided example file, note that
   the following sections have to be defined:

   - `H_fname`, giving the filename of matrix elements of the chosen Hamiltonian
     in the `numpy` format. Four such files are already provided in the
     directory [matrices](https://github.com/js216/State-evolution/tree/master/matrices)

   - `scan_param` and `scan_range`, giving the parameter that is to be scanned
     over

   - `fixed_params`, giving the names and values of the fixed parameters that
     are to be passed to the `field` function, as well as displayed in the plots

   - `time_params` are used by `time_mesh()` together with all the field
     parameters to determine how fine a time mesh to make

   - `state_idx` determines which state's  in the enumeration of eigenstates of
     `H(t=field(t_final))`

3. Run the `main.py` script, giving it the run directory, and the `.json` file
   as arguments. For example:

       python3 src/main.py runs/example/ example.json

   The program prints the number of time steps to be used with the chosen time
   mesh, and the progress of the calculation (using `tqdm`).

### Parallel evaluation

The most time-consuming part of the calculation, matrix exponentiation, is
already parallelized by
[scipy.linalg.expm](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.linalg.expm.html)
in a multi-threaded way. Nonetheless, each scan with a decent number of steps
will take hours on a modern machine. Thus, I recommend defining multiple option
files within a single `run_dir`, and running each scan on a separated node in a
cluster. Moreover, if the main script uses all available MPI ranks. Refer to the
[YCRC instructions](https://docs.ycrc.yale.edu/clusters-at-yale/) to learn how
to access the cluster at Yale.
