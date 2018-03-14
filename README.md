# KinematicModelling


Python modules used to estimate centre velocity, velocity dispersion and parallaxes for open clusters, based on Lindegren et al. (2000, http://adsabs.harvard.edu/abs/2000A%26A...356.1119L). 


Together with Stella Reino, we also developed a procedure that includes radial velocitie. More details on this modified version 
of the code can be found in Reino et al. (2018, subm., minor revision).

The version of the method *without* the inclusion of radial velocities was applied in Bravi et al. (2018) (http://adsabs.harvard.edu/abs/2018arXiv180301908B).

## Prerequisites

The routines depend on the following python packages:
1. PyGaia
2. astropy
3. numpy, scipy;
4. emcee;
5. coner.


## Notes
The notebooks provide an example for a possible application of the routines in routines_mod.py and derivatives_mod.py.
The space motion is derived for the nearby open cluster IC 2602.

The steps of the procedure are explained in the notebooks (I hope clearly enough!).




