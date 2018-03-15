import numpy as np
from pygaia.astrometry.vectorastrometry import phaseSpaceToAstrometry, normalTriad
from pygaia.astrometry.constants import *
from matrix_operations import *
from covmatrix import *
import scipy.optimize as opt
from  scipy.linalg import det, inv
from scipy.stats import chi2


_A = auKmYearPerSec  ### km*yr/s

def _like4(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, i):
	"""
	Estimate the likelihood function for every set of observation.
	Then, estimate the function U(init_par) that needs to be minimized 
	to find the best fit to the parameters.

	Parameters:
	------------
	init_par - Set of initial values for: 1) i-th parallax [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta -  position of i-th star [rad];
	plx_obs, mualpha_obs, mudelta_obs -  observed values for parallaxes and proper motions [mas, mas/yr] of i-th star;
	vrad_obs -  radial velocities [km/s] of i-th star;

	sigma_obs - 3-dim array of observed errors for parallax and proper motion [mas, mas/yr] of i-th star;
	sigma_vrad -  error on radial velocities [km/s] of i-th star;

	ccoef - 3-dim array of correlation coefficients from the HIP catalogue of i-th star. 
	
	Returns:
	-----------
	g - values of g_i(theta) for the i-th star, see eq. 19 in Lindegren+2000;
	U(init_par) - The function in Eq. (18) of Lindegren+2000, for the i-th star.
	"""
	
	plx_mod, v, sigma_v = init_par[i], init_par[-4:-1], init_par[-1]  
	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*plx_mod/_A ### [mas/yr]
	mudelta_mod = np.dot(np.transpose(q),v)*plx_mod/_A ### [mas/yr]
	### Add the model vector for the radial velocities:
	vrad_mod = np.dot(np.transpose(r),v)  ### [km/s]
  	
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)
	C = np.zeros((4,4),dtype=np.float64) ### This is a 4x4 matrix 
	### Diagonal terms:
	C[0,0],C[1,1],C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	C[3,3] = sigma_vrad**2.
	
	r_plx_muRa, r_plx_muDec, r_muRa_muDec = ccoef[0], ccoef[1], ccoef[2] 
 
	### Correlation terms:
	C[0,1], C[0,2] = r_plx_muRa*sigma_plx*sigma_mualpha, r_plx_muDec*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = r_plx_muRa*sigma_plx*sigma_mualpha, r_muRa_muDec*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = r_plx_muDec*sigma_plx*sigma_mudelta, r_muRa_muDec*sigma_mualpha*sigma_mudelta

	E = np.zeros((4,4),dtype=np.float64) ### 4x4 matrix 
	E[1,1],E[2,2] = (sigma_v**2.)*(plx_mod/_A)**2., (sigma_v**2.)*(plx_mod/_A)**2.     ### [mas/yr]
	E[3,3] = sigma_v**2.								   ### [km/s]

	
	D = np.add(E,C)
	detD =  det(D) 
	invD =  inv(D)
		
	a_c = np.array([plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs-mudelta_mod, vrad_obs - vrad_mod])
	g_func = row_matrix_col_4d(a_c, a_c, invD) 
	
	
	return  detD, g_func

def _like3(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, sigma_obs, ccoef, i):
	"""
	Estimate the likelihood function for every set of observation.
	Then, estimate the function U(init_par) that needs to be minimized 
	to find the best fit to the parameters.

	Parameters:
	------------
	init_par - Set of initial values for: 1) i-th parallax [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta -  position of i-th star [rad];
	plx_obs, mualpha_obs, mudelta_obs -  observed values for parallaxes and proper motions [mas, mas/yr] of i-th star;
	vrad_obs -  radial velocities [km/s] of i-th star;

	sigma_obs - 3-dim array of observed errors for parallax and proper motion [mas, mas/yr] of i-th star;
	sigma_vrad -  error on radial velocities [km/s] of i-th star;

	ccoef - 3-dim array of correlation coefficients from the HIP catalogue of i-th star. 
	
	Returns:
	-----------
	g - values of g_i(theta) for the i-th star, see eq. 19 in Lindegren+2000;
	U(init_par) - The function in Eq. (18) of Lindegren+2000, for the i-th star.	
	"""

	plx_mod, v, sigma_v =  init_par[i], init_par[-4:-1], init_par[-1]  
	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*plx_mod/_A
	mudelta_mod = np.dot(np.transpose(q),v)*plx_mod/_A
  	
	sigma_plx, sigma_mualpha, sigma_mudelta = sigma_obs
	r_plx_muRa, r_plx_muDec, r_muRa_muDec = ccoef[0], ccoef[1], ccoef[2] 
	
	C = np.zeros((3,3),dtype=np.float64)
	C[0,0],C[1,1],C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	C[0,1], C[0,2] = r_plx_muRa*sigma_plx*sigma_mualpha, r_plx_muDec*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = r_plx_muRa*sigma_plx*sigma_mualpha, r_muRa_muDec*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = r_plx_muDec*sigma_plx*sigma_mudelta, r_muRa_muDec*sigma_mualpha*sigma_mudelta

	E = np.zeros((3,3),dtype=np.float64)
	E[1,1],E[2,2] = (sigma_v**2.)*(plx_mod/_A)**2., (sigma_v**2.)*(plx_mod/_A)**2.
	
	D = np.add(E,C)
	detD =  det(D) 
	invD =  inv(D)
		
	a_c = np.array([plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs-mudelta_mod])
	g_func = row_matrix_col(a_c, a_c, invD)
	
	
	return  detD, g_func




def ilike(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):
	"""
	Estimate the function U (exponent of the likelihood), which needs to be minimized.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - the number of stars;

	Returns:
	-----------
	L - function to be minimized, see eq. 18 in Lindegren+2000;
	g_func - array with the values of the function g_i(theta)

	"""
	detD, gfunc, U  = np.zeros(N), np.zeros(N), np.zeros(N)
	for i in range(N):
		if np.isfinite(vrad_obs[i]):
			detD[i], gfunc[i] =  _like4(init_par, alpha[i], delta[i], plx_obs[i],
						 mualpha_obs[i], mudelta_obs[i], vrad_obs[i], sigma_obs[i,:], sigma_vrad[i], ccoef[i, :], i)
			U[i] = np.log(detD[i]) + gfunc[i] + 4.*np.log(2.*np.pi)
			
		else:
			detD[i], gfunc[i] = _like3(init_par, alpha[i], delta[i], plx_obs[i],
						 mualpha_obs[i], mudelta_obs[i], sigma_obs[i,:], ccoef[i, :], i)	
			U[i] = np.log(detD[i]) + gfunc[i] + 3.*np.log(2.*np.pi)

	L = np.sum(U)
	return  L, gfunc

def Ulike(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):
	"""
	Returns the function to be minimized ---> this needs to be called in the main.  
	"""

	L, g = ilike(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N)
	return L


def g_func(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):
	"""
	Estimate the g_func (exponent of the likelihood), which gives an estimate
	of the membership of each star to the moving group.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - the number of stars;

	Returns:
	-----------
	g_func - An array of values with the values of g_i(theta) for each star in the group, see eq. 19 in Lindegren+2000;
	"""
	L, g = ilike(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N) 	
	p = np.zeros(N)
	for i in range(N):
	    if np.isfinite(vrad_obs[i]):
	        p[i] = chi2.sf(g[i],3)
	    else:
	        p[i] = chi2.sf(g[i],2)
	        
	return  p

def optimizer(grad, method, init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoeff, N):
	"""
	Minimize the likelihood function using different methods.
	
	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	vrad_obs, sigma_vrad - observed radial velocities and observed errors; if the radial velocity of one particular
			       object is not measured, use the value -9999.99 for that star.
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - the number of stars;
	
	Returns: 
	res.x - An array containing the results of the minimization.

	"""

	
	if grad == 'NO':
		if method == 'Powell' :
			res = opt.minimize(Ulike,init_par,  method = method,
			           args = (alpha, delta, plx_obs, mualpha_obs,mudelta_obs, vrad_obs, sigma_obs, sigma_vrad,  ccoeff, N))
			return res.x, res.nit
		elif method == 'Nelder-Mead':
			res = opt.minimize(Ulike,init_par, method = method,
			           args = (alpha, delta, plx_obs, mualpha_obs,mudelta_obs, vrad_obs, sigma_obs, sigma_vrad,  ccoeff, N),
				   options = {'ftol': 0.0001})
			return res.x, res.nit
		elif method == 'default':
			res = opt.minimize(Ulike,init_par, 
			           args = (alpha, delta, plx_obs, mualpha_obs,mudelta_obs, vrad_obs, sigma_obs, sigma_vrad,  ccoeff, N))
			return res.x, res.nit

	elif grad == 'YES':
		res = opt.minimize(Ulike, init_par, method = method, jac = stella_grad_full, 
 			   args = (alpha, delta, plx_obs, mualpha_obs,mudelta_obs, vrad_obs, sigma_obs, sigma_vrad,  ccoeff, N),
			   options={'disp': True, 'maxiter': 4000, 'xtol': 1e-4})
		return res.x, res.nit 
			
		
	elif grad == 'HESS':
		res = opt.minimize(Ulike, init_par, method = method, jac = stella_grad_full, hess = stella_hessian,
					   args = (alpha, delta, plx_obs, mualpha_obs,mudelta_obs, vrad_obs, sigma_obs, sigma_vrad,  ccoeff, N),
					   options = {'disp': True, 'maxiter': 4000, 'xtol': 1.e-06}) 
		return res.x, res.nit

		



