These codes accompany the manuscript: 

Reitman, Nadine G., Mueller, KJ, Tucker, GE, Gold, RD, Briggs RW, and Barnhart, KR 
Offset channels do not accurately record strike-slip fault displacement: Evidence from landscape evolution models

Submitted for review in August 2019.

Please cite the above manuscript as well as this Zenodo repository if using these codes for research.

All codes written by Nadine Reitman
Send questions to: nadine.reitman@colorado.edu

This repository contains:
• config.yaml
Example parameter input file to run strike_slip_model.py. Set all parameter values here.

• config_files.zip
Contains all config files for all model runs analyzed in Reitman et al. manuscript.

• slip.zip
Contains functions in slip.py required to run strike_slip_model.py. make this a function by running “python setup.py develop” from inside the slip directory. if running in a conda environment, do this in each conda environment you run strike_slip_model.py in. 

• strike_slip_model.py
The landscape evolution model. run on the command line using:  “python strike_slip_model.py config.yaml”

• drainage_area.py
Code to calculate drainage area from model topography and save it as an ASCII file

• measure_offsets.py
Code with a function to measure offsets from output from the strike-slip model, as well as example implementation for 1 model. requires an ASCII file of drainage area.

• parameter_values.xlsx
Table of all parameter values used in model runs in Reitman et al. manuscript. Model name here corresponds to naming convention for the config files.

• ztopo_1000x500y1.asc
Initial topography grid ASCII file used in all model runs in Reitman et al. manuscript. If importing this grid file into Landlab to seed a landscape evolution model run, the dimensions are 1000 cells in x-direction by 500 cells in y-direction, with 1 m spacing.


The steps to successfully execute the model and measure channel offsets are:
0) make sure functions in slip.py are installed correctly
1) set parameters in config.yaml
2) run strike_slip_model.py (optionally, use ztopo_1000x500y1.asc as initial topography)
3) run drainage_area.py on the output model topography ASCII file
4) run measure_offsets.py on the drainage area ASCII file

All codes were written in Python3.

Package dependencies:
•numpy	
•matplotlib
•scipy
•scikit-image
•scikit-learn
•pickle
•yaml
•rasterio
•landlab

Send questions to: nadine.reitman@colorado.edu
