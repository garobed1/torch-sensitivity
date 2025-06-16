# torch-sensitivity

Repository for the development of multi-fidelity torch uncertainty quantification methods and analysis

We recommend installing this repository as a package by running:
```
pip3.11 install -e .
```
at the top level directory, since it contains useful utility files for the entire torch UQ pipeline. 

We also recommend symbolically linking torch1d.py at the top level directory for execution of samples
```
ln -s /path/to/torch1d/torch1d.py .
```

## Directories

`tuq_main`: Contains the primary UQ pipeline scripts
`tuq_misc`: Miscellaneous scripts for examining sample runs and meshes
`tuq_util`: Useful utility classes and functions


## Main UQ Pipelines in `tuq_main`

Scripts to automate/parallelize these processes for many samples are in
`dane_scripts`

### Building reaction rate KL models and sampling them

```
resample_rates.py
```

### Sampling Torch1D

```
rate_data_to_torch1d_input.py
run_torch_samples.py
```
Post-processing
```
sf_torch1d_uq_post.py
```
Optional
```
compare_qoi_dists.py
reaction_sensitivity.py
reaction_sens_plots.py
```

### Sampling TPS (Axisymmetric)

```
rate_data_to_tps2d_input.py
manage_bigcomp_runs.py
manage_bigcomp_runs.py --run
```
Post-processing
```
sf_tps2d_uq_post.py
```