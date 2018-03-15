### Purpose: compute the likelihood function that we want to minimize.
### The functions defined here are (detailed description are 
### present  at the beginnin of each function):
### 1) _like();
### 2) Ulike;
### 3) g_func;  
### 4) optimizer.
### E.Z., 04-07-2016.

import numpy as np
from pygaia.astrometry.vectorastrometry import phaseSpaceToAstrometry, normalTriad
from pygaia.astrometry.constants import *
from matrixOperations import *
from derivatives_mod import *
import scipy.optimize as opt
from  scipy.linalg import det, inv


_A = auKmYearPerSec  ### km*yr/s

def matrix_det_inv(mat):
    """
    Calculate the determinant and inverse of a 3x3 matrix. This code is specialized for the 3x3 case and
    hopefully faster than standard python functions.

    NOTE that no precautions are taken to ensure numerically stable calculations. The code is just a
    straightforward implementation of the formal mathematical inverse of a 3x3 matrix.

    Parameters
    ----------

    mat - input 3x3 matrix

    Returns
    -------

    determinant, inverse matrix
    """
    a = mat[0,0]
    b = mat[0,1]
    c = mat[0,2]
    d = mat[1,0]
    e = mat[1,1]
    f = mat[1,2]
    g = mat[2,0]
    h = mat[2,1]
    i = mat[2,2]

    det = b*f*g + c*d*h + a*e*i - a*f*h - b*d*i - c*e*g

    invmat = np.zeros((3,3))
    invmat[0,0] = (e*i - f*h) / det
    invmat[0,1] = (c*h - b*i) / det
    invmat[0,2] = (b*f - c*e) / det
    invmat[1,0] = (f*g - d*i) / det
    invmat[1,1] = (a*i - c*g) / det
    invmat[1,2] = (c*d - a*f) / det
    invmat[2,0] = (d*h - e*g) / det
    invmat[2,1] = (b*g - a*h) / det
    invmat[2,2] = (a*e - b*d) / det

    return det, invmat

def _like(init_par, alpha, delta, obs, sigma_obs, ccoef, N):
	"""
	Estimate the likelihood function for every set of observations.
	Then, estimate the function U(init_par) that needs to be minimized 
	to find the best fit to the parameters.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Array of cluster member positions [rad];
	obs - Matrix  of observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - 3-dim array of observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients. 
	N - the number of stars;

	Returns:
	-----------
	g - An array of values with the values of g_i(theta) for each star in the group, see eq. 19 in Lindegren+2000;
	U(init_par) - The function in Eq. (18) of Lindegren+2000, i.e. the function that needs to be minimized.
	"""
	
	plx_mod, v, sigma_v = init_par[:-4], init_par[-4:-1], init_par[-1] 
	plx_obs, mualpha_obs, mudelta_obs = obs[:, 0], obs[:, 1], obs[:, 2]
 
	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*plx_mod/_A
	mudelta_mod = np.dot(np.transpose(q),v)*plx_mod/_A
  	
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)
	
	C = np.zeros((3,3,N),dtype=np.float64)
	C[0,0,:],C[1,1,:],C[2,2,:] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
 
	r_plx_muRa, r_plx_muDec, r_muRa_muDec = np.zeros(N), np.zeros(N), np.zeros(N)
	r_plx_muRa[:], r_plx_muDec[:], r_muRa_muDec[:] = ccoef[:, 0], ccoef[:, 1], ccoef[:, 2] 
	C[0,1,:], C[0,2,:] = r_plx_muRa*sigma_plx*sigma_mualpha, r_plx_muDec*sigma_plx*sigma_mudelta
	C[1,0,:], C[1,2,:] = r_plx_muRa*sigma_plx*sigma_mualpha, r_muRa_muDec*sigma_mualpha*sigma_mudelta
	C[2,0,:], C[2,1,:] = r_plx_muDec*sigma_plx*sigma_mudelta, r_muRa_muDec*sigma_mualpha*sigma_mudelta

	E = np.zeros((3,3,N),dtype=np.float64)
	E[1,1,:],E[2,2,:] = (sigma_v**2.)*(plx_mod/_A)**2., (sigma_v**2.)*(plx_mod/_A)**2.
	D,invD, detD  = np.zeros((3,3,N),dtype=np.float64),np.zeros((3,3,N),dtype=np.float64), np.ones(N)
	D = np.add(E,C)
	
	for i in range(N):
		det, invmat = matrix_det_inv(D[:, :, i])
		#print(det, invmat)
		detD[i] =  det 
		invD[:,:,i] =  invmat
		

	a_c = np.ones((3,N))
	a_c = [plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs-mudelta_mod]

	g_func = row_matrix_col(a_c, a_c, invD)
	
	like = np.ones(N)
	like = ((2*np.pi)**(-1.5)*detD**(-0.5))*np.exp(-0.5*g_func)

	return  np.array(np.sum(np.log(detD))+ np.sum(g_func)), g_func



def Ulike(init_par, alpha, delta, obs, sigma_obs, ccoef, N):
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
	ccoef - 3-dim array of correlation coefficients;
	N - the number of stars;

	Returns:
	-----------
	U - An array of values with the values of g_i(theta) for each star in the group, see eq. 18 in Lindegren+2000;
	"""
	U, g = _like(init_par, alpha, delta, obs, sigma_obs, ccoef, N) 	

	return  U


def g_func(init_par, alpha, delta, obs, sigma_obs, ccoef, N):
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
	ccoef - 3-dim array of correlation coefficients;
	N - the number of stars;

	Returns:
	-----------
	g_func - An array of values with the values of g_i(theta) for each star in the group, see eq. 19 in Lindegren+2000;
	"""
	U, g = _like(init_par, alpha, delta, obs, sigma_obs, ccoef, N) 	

	return  g

def optimizer(grad, method, init_par, alpha, delta, obs, sigma_obs, ccoeff, N):
	"""
	Minimize the likelihood function specifying the method that you want to use.
	
	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients;
	N - the number of stars;
	
	Returns: 
	res.x - An array containing the the parameters resulting from the minimization.

	"""

	
	if grad == 'NO':
		if method == 'Powell' :
			res = opt.minimize(Ulike,init_par,  method = method,
			           args = (alpha, delta, obs, sigma_obs, ccoeff, N),
				    options = {'disp':True,'ftol': 1.0e-2, 'maxfev': 30000})
			return res.x
		elif method == 'Nelder-Mead':
			res = opt.minimize(Ulike,init_par, method = method,
			           args = (alpha, delta, obs, sigma_obs, ccoeff, N),
				   options = {'disp':True,'ftol': 1.0e-6, 'maxfev': 1000000})
			return res.x
		elif method == 'default':
			res = opt.minimize(Ulike,init_par, 
			           args = (alpha, delta, obs, sigma_obs, ccoeff, N))
			return res.x

	elif grad == 'YES':
		res = opt.minimize(Ulike, init_par, method = method, jac = gradient, 
 			   args = (alpha, delta, obs, sigma_obs, ccoeff, N),
			   options={'disp': True, 'maxiter': 4000, 'xtol': 1e-4}) 
		return res.x
	elif grad == 'HESS':
		res = opt.minimize(Ulike, init_par, method = method, jac = gradient, hess = hessian,
					   args = (alpha, delta, obs, sigma_obs, ccoeff, N),
					   options = {'disp': True, 'maxiter': 4000, 'xtol': 1.e-06}) 
		
		return res.x



