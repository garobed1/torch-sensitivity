import os

"""
Script to produce torch1d input files from reaction rate sample data

Options include a template input file and a directory containing the rate
samples in .h5 format

"""


template_file = "torch1d_argon_sample_config_template.yml"
sample_dir = "../torch-chemistry/argon/results/sevenSpecies"