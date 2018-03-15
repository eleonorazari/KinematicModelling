import numpy as np
from matrix_operations import *
from pygaia.astrometry.vectorastrometry import phaseSpaceToAstrometry, normalTriad
from pygaia.astrometry.constants import *

_A = auKmYearPerSec

def stella_grad_full(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):

	f = np.zeros(N+4)

	for i in range(N):
		if np.isfinite(vrad_obs[i]):
			f += stella_grad_4d(init_par, alpha[i], delta[i], plx_obs[i], mualpha_obs[i], mudelta_obs[i], vrad_obs[i], sigma_obs[i,:], sigma_vrad[i], ccoef[i,:], N, i)			

		elif np.isnan(vrad_obs[i]):
			f += stella_grad_3d(init_par, alpha[i], delta[i], plx_obs[i], mualpha_obs[i], mudelta_obs[i], vrad_obs[i], sigma_obs[i,:], sigma_vrad[i], ccoef[i,:], N, i)
	return f

def stella_grad_3d(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N, i):
	"""
	Estimate the jacobian matrix of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				     	      2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       	      3) The cluster velocity dispersion [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	"""
       
	parallax, v, sigma_v = init_par[i], init_par[-4:-1], init_par[-1] 

	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A

	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)

	C = np.zeros((3,3),dtype=np.float64)
	C[0,0], C[1,1], C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.

	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta = ccoef[0], ccoef[1], ccoef[2] 
	
	C[0,1], C[0,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	E = np.zeros((3,3),dtype=np.float64)
	E[1,1], E[2,2]= (sigma_v**2.)*(parallax/_A)**2., (sigma_v**2.)*(parallax/_A)**2.
	
	
	D, invD = np.zeros((3,3),dtype = np.float64),np.zeros((3,3),dtype=np.float64)
	D = np.add(E,C)
	
	invD = np.linalg.inv(D)
	
	a_c = np.array([plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs - mudelta_mod])

	
	cprime_pi, cprime_vx, cprime_vy, cprime_vz,  = np.ones(3), np.ones(3), np.ones(3), np.ones(3)
	
	cprime_pi[0] = 1.
	cprime_pi[1] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2] = np.dot(np.transpose(q),v)/_A
	
	cprime_vx[0] = 0.
	cprime_vx[1] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A

	
	cprime_vy[0] = 0.
	cprime_vy[1] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A


	cprime_vz[0] = 0.
	cprime_vz[1] = 0. 
	cprime_vz[2] = np.cos(delta)*plx_mod/_A


	dD_dpi, dD_dsigmav = np.zeros((3,3)),  np.zeros((3,3))
	
	dD_dpi[1,1] = 2.*plx_mod*((sigma_v/_A)**2.)
	dD_dpi[2,2] = 2.*plx_mod*((sigma_v/_A)**2.)

	dD_dsigmav[1,1] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[2,2] = 2.*sigma_v*((plx_mod/_A)**2.)


	### See formula A.3
	f = np.zeros(N+4)
	
	
	f_pi =  0
	f_vx, f_vy, f_vz, f_sigmav = 0, 0, 0, 0

	f_vx += np.dot(np.dot(invD, cprime_vx), a_c)
	f_vy += np.dot(np.dot(invD, cprime_vy), a_c)
	f_vz += np.dot(np.dot(invD, cprime_vz), a_c)
	f_pi += np.dot(np.dot(invD, cprime_pi), a_c)	

	f_pi -= 0.5 * np.trace(np.dot(invD, dD_dpi))
	f_pi += 0.5 * np.dot(np.dot(np.dot(np.dot(invD, dD_dpi), invD), a_c), a_c)
				
	f_sigmav -= 0.5 * np.trace(np.dot(invD, dD_dsigmav))
	f_sigmav += 0.5 * np.dot(np.dot(np.dot(np.dot(invD, dD_dsigmav), invD), a_c), a_c)
		
	f[i] = f_pi
		
	f[-4], f[-3], f[-2], f[-1] = f_vx, f_vy, f_vz, f_sigmav
		
	### f is Grad L(theta), see Eq. 17
	return -2*f   ### Grad U(theta), see Eq. 18

def stella_grad_4d(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N, i):
	"""
	Estimate the jacobian matrix of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				     	      2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       	      3) The cluster velocity dispersion [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	"""
       
	parallax, v, sigma_v = init_par[i], init_par[-4:-1], init_par[-1] 

	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A
	vrad_mod = np.dot(np.transpose(r),v)

	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)

	C = np.zeros((4,4),dtype=np.float64)
	C[0,0], C[1,1], C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	C[3,3] = sigma_vrad**2.
	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta = ccoef[0], ccoef[1], ccoef[2] 
	
	C[0,1], C[0,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	E = np.zeros((4,4),dtype=np.float64)
	E[1,1], E[2,2], E[3,3] = (sigma_v**2.)*(parallax/_A)**2., (sigma_v**2.)*(parallax/_A)**2., sigma_v**2.
	
	
	D, invD = np.zeros((4,4),dtype = np.float64),np.zeros((4,4),dtype=np.float64)
	D = np.add(E,C)
	
	invD = np.linalg.inv(D)
	
	a_c = np.array([plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs - mudelta_mod, vrad_obs - vrad_mod])
	
	cprime_pi, cprime_vx, cprime_vy, cprime_vz  = np.ones(4), np.ones(4), np.ones(4), np.ones(4)
	
	cprime_pi[0] = 1.
	cprime_pi[1] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2] = np.dot(np.transpose(q),v)/_A
	cprime_pi[3] = 0.
	
	cprime_vx[0] = 0.
	cprime_vx[1] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A
	cprime_vx[3] = np.cos(delta)*np.cos(alpha)

	cprime_vy[0] = 0.
	cprime_vy[1] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A
	cprime_vy[3] = np.cos(delta)*np.sin(alpha)

	cprime_vz[0] = 0.
	cprime_vz[1] = 0. 
	cprime_vz[2] = np.cos(delta)*plx_mod/_A
	cprime_vz[3] = np.sin(delta)

	dD_dpi, dD_dsigmav = np.zeros((4,4)),  np.zeros((4,4))
	
	dD_dpi[1,1] = 2.*plx_mod*((sigma_v/_A)**2.)
	dD_dpi[2,2] = 2.*plx_mod*((sigma_v/_A)**2.)

	dD_dsigmav[1,1] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[2,2] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[3,3] = 2.*sigma_v

	### See formula A.3
	f = np.zeros(N+4)
	
	
	f_pi =  0
	f_vx, f_vy, f_vz, f_sigmav = 0, 0, 0, 0

	f_vx += np.dot(np.dot(invD, cprime_vx), a_c)
	f_vy += np.dot(np.dot(invD, cprime_vy), a_c)
	f_vz += np.dot(np.dot(invD, cprime_vz), a_c)
	f_pi += np.dot(np.dot(invD, cprime_pi), a_c)	

	f_pi -= 0.5 * np.trace(np.dot(invD, dD_dpi))
	f_pi += 0.5 * np.dot(np.dot(np.dot(np.dot(invD, dD_dpi), invD), a_c), a_c)
				
	f_sigmav -= 0.5 * np.trace(np.dot(invD, dD_dsigmav))
	f_sigmav += 0.5 * np.dot(np.dot(np.dot(np.dot(invD, dD_dsigmav), invD), a_c), a_c)
		
	f[i] = f_pi

	f[-4], f[-3], f[-2], f[-1] = f_vx, f_vy, f_vz, f_sigmav
		
	### f is Grad L(theta), see Eq. 17
	return -2*f   ### Grad U(theta), see Eq. 18



def stella_Nmatrix_full(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):

	Nmatrix = np.zeros((N+4, N+4))

	for i in range(N):
		if np.isfinite(vrad_obs[i]):
			Nmatrix += stella_Nmatrix_4d(init_par, alpha[i], delta[i], plx_obs[i], mualpha_obs[i], mudelta_obs[i], vrad_obs[i], sigma_obs[i,:], sigma_vrad[i], ccoef[i,:], N, i)			

		elif np.isnan(vrad_obs[i]):
			Nmatrix += stella_Nmatrix_3d(init_par, alpha[i], delta[i], plx_obs[i], mualpha_obs[i], mudelta_obs[i], vrad_obs[i], sigma_obs[i,:], sigma_vrad[i], ccoef[i,:], N, i)
	return Nmatrix
	
def stella_Nmatrix_3d(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N, i):
	"""
	Estimate covariance matrix of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				     	      2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       	      3) The cluster velocity dispersion [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	Returns:
	-----------
	Cov - N+4 x N+4 matrix, corresponding to the covariance of the parameters of the likelihood function.
	"""
	parallax, v, sigma_v = init_par[i], init_par[-4:-1], init_par[-1] 

	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A

	
	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)

	C = np.zeros((3,3),dtype=np.float64)
	C[0,0],C[1,1],C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.

	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta = ccoef[0], ccoef[1], ccoef[2] 
	
	C[0,1], C[0,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	E = np.zeros((3,3),dtype=np.float64)
	E[1,1], E[2,2]= (sigma_v**2.)*(parallax/_A)**2., (sigma_v**2.)*(parallax/_A)**2.
	D, invD = np.zeros((3,3),dtype=np.float64),np.zeros((3,3),dtype=np.float64)
	D = np.add(E,C)
	
	invD = np.linalg.inv(D)
	
	
	cprime_pi, cprime_vx, cprime_vy, cprime_vz,  = np.ones(3), np.ones(3), np.ones(3), np.ones(3)
	cprime_pi[0] = 1.
	cprime_pi[1] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2] = np.dot(np.transpose(q),v)/_A
	
	cprime_vx[0] = 0.
	cprime_vx[1] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A

	
	cprime_vy[0] = 0.
	cprime_vy[1] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A

	cprime_vz[0] = 0.
	cprime_vz[1] = 0. 
	cprime_vz[2] = np.cos(delta)*plx_mod/_A
	
	dD_dpi, dD_dsigmav, dD_dpi2, dD_dsigmav2, dD_dpisigmav = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
	
	dD_dpi[1,1] = 2.*plx_mod*((sigma_v/_A)**2.)
	dD_dpi[2,2] = 2.*plx_mod*((sigma_v/_A)**2.)

	dD_dsigmav[1,1] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[2,2] = 2.*sigma_v*((plx_mod/_A)**2.)
	
	dD_dpi2[1,1] = 2.*((sigma_v/_A)**2.)
	dD_dpi2[2,2] = 2.*((sigma_v/_A)**2.)
	
	dD_dsigmav2[1,1] = 2.*((plx_mod/_A)**2.)
	dD_dsigmav2[2,2] = 2.*((plx_mod/_A)**2.)
 
	dD_dpisigmav[1,1] = 4.*plx_mod*sigma_v/((_A)**2.)
	dD_dpisigmav[2,2] = 4.*plx_mod*sigma_v/((_A)**2.)	

	
	### See formula A.7
	hess = np.zeros((N+4, N+4))
		
	hess_pi2, hess_vx2, hess_vy2, hess_vz2, hess_sigmav2 = 0, 0, 0, 0, 0
	hess_pi_vx, hess_pi_vy, hess_pi_vz, hess_pi_sigmav = 0, 0, 0 , 0
	hess_vx_vy, hess_vx_vz, hess_vy_vz = 0, 0, 0
	
	
	hess_vx2 += np.dot(np.dot(invD, cprime_vx), cprime_vx)
	hess_vy2 += np.dot(np.dot(invD, cprime_vy), cprime_vy)
	hess_vz2 += np.dot(np.dot(invD, cprime_vz), cprime_vz)
	hess_pi2 += np.dot(np.dot(invD, cprime_pi), cprime_pi)
	hess_pi_vx += np.dot(np.dot(invD, cprime_pi), cprime_vx)
	hess_pi_vy += np.dot(np.dot(invD, cprime_pi), cprime_vy)
	hess_pi_vz += np.dot(np.dot(invD, cprime_pi), cprime_vz)
	hess_vx_vy += np.dot(np.dot(invD, cprime_vx), cprime_vy)
	hess_vx_vz += np.dot(np.dot(invD, cprime_vx), cprime_vz)
	hess_vy_vz += np.dot(np.dot(invD, cprime_vy), cprime_vz)
		


	hess_pi2 += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dpi)) + np.dot(invD,dD_dpi2))
	hess_pi2 += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dpi)), invD) - np.dot(np.dot(invD,dD_dpi2), invD))
				
	hess_sigmav2 += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dsigmav), np.dot(invD, dD_dsigmav)) + np.dot(invD,dD_dsigmav2))
	hess_sigmav2 += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dsigmav), np.dot(invD, dD_dsigmav)), invD) - np.dot(np.dot(invD, dD_dsigmav2), invD))
		
	hess_pi_sigmav += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dsigmav)) + np.dot(invD,dD_dpisigmav))
	hess_pi_sigmav += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dsigmav)), invD) - np.dot(np.dot(invD, dD_dpisigmav), invD))
	
	
	hess[i,i] = hess_pi2
	hess[i,-1] = hess_pi_sigmav
	hess[i,-2] = hess_pi_vz 
	hess[i,-3] = hess_pi_vy
	hess[i,-4] = hess_pi_vx
	hess[-1,i] = hess[i,-1]
	hess[-2,i] = hess[i,-2]
	hess[-3,i] = hess[i,-3]
	hess[-4,i] = hess[i,-4]
		
	hess[-1,-1], hess[-2,-2], hess[-3,-3], hess[-4,-4] = hess_sigmav2, hess_vz2, hess_vy2, hess_vx2 
	hess[-2,-3], hess[-3,-4], hess[-2,-4] = hess_vy_vz, hess_vx_vy, hess_vx_vz
	hess[-3,-2], hess[-4,-3], hess[-4,-2] = hess[-2,-3], hess[-3,-4], hess[-2,-4]	
	
	return hess ### N = -E(H) for L
	
def stella_Nmatrix_4d(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N, i):
	"""
	Estimate covariance matrix of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				     	      2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       	      3) The cluster velocity dispersion [km/s];
	alpha, delta - Cluster member positions [rad];
	plx_obs, mualpha_obs, mudelta_obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	Returns:
	-----------
	Cov - N+4 x N+4 matrix, corresponding to the covariance of the parameters of the likelihood function.
	"""
	
	parallax, v, sigma_v = init_par[i], init_par[-4:-1], init_par[-1]  

	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A
	vrad_mod = np.dot(np.transpose(r),v)
	
	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)

	C = np.zeros((4,4),dtype=np.float64)
	C[0,0],C[1,1],C[2,2] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	C[3,3] = sigma_vrad**2.
	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta= ccoef[0], ccoef[1], ccoef[2] 
	
	C[0,1], C[0,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0], C[1,2] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0], C[2,1] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	E = np.zeros((4,4),dtype=np.float64)
	E[1,1],E[2,2], E[3,3] = (sigma_v**2.)*(parallax/_A)**2., (sigma_v**2.)*(parallax/_A)**2., sigma_v**2.
	D, invD = np.zeros((4,4),dtype=np.float64),np.zeros((4,4),dtype=np.float64)
	D = np.add(E,C)
	
	invD = np.linalg.inv(D)
	
	
	cprime_pi, cprime_vx, cprime_vy, cprime_vz,  = np.ones(4), np.ones(4), np.ones(4), np.ones(4)
	
	cprime_pi[0] = 1.
	cprime_pi[1] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2] = np.dot(np.transpose(q),v)/_A
	cprime_pi[3] = 0
	
	cprime_vx[0] = 0.
	cprime_vx[1] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A
	cprime_vx[3] =  np.cos(delta)*np.cos(alpha)

	
	cprime_vy[0] = 0.
	cprime_vy[1] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A
	cprime_vy[3] = np.cos(delta)*np.sin(alpha)

	cprime_vz[0] = 0.
	cprime_vz[1] = 0. 
	cprime_vz[2] = np.cos(delta)*plx_mod/_A
	cprime_vz[3] = np.sin(delta)

	dD_dpi, dD_dsigmav, dD_dpi2, dD_dsigmav2, dD_dpisigmav  = np.zeros((4,4)),  np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))
	
	dD_dpi[1,1] = 2.*plx_mod*((sigma_v/_A)**2.)
	dD_dpi[2,2] = 2.*plx_mod*((sigma_v/_A)**2.)

	dD_dsigmav[1,1] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[2,2] = 2.*sigma_v*((plx_mod/_A)**2.)
	dD_dsigmav[3,3] = 2.*sigma_v
	
	dD_dpi2[1,1] = 2.*((sigma_v/_A)**2.)
	dD_dpi2[2,2] = 2.*((sigma_v/_A)**2.)
	
	dD_dsigmav2[1,1] = 2.*((plx_mod/_A)**2.)
	dD_dsigmav2[2,2] = 2.*((plx_mod/_A)**2.)
	dD_dsigmav2[3,3] = 2.
 
	dD_dpisigmav[1,1] = 4.*plx_mod*sigma_v/((_A)**2.)
	dD_dpisigmav[2,2] = 4.*plx_mod*sigma_v/((_A)**2.)	

	
	### See formula A.7
	hess = np.zeros((N+4, N+4))
		
	hess_pi2, hess_vx2, hess_vy2, hess_vz2, hess_sigmav2 = 0, 0, 0, 0, 0
	hess_pi_vx, hess_pi_vy, hess_pi_vz, hess_pi_sigmav = 0, 0, 0, 0
	hess_vx_vy, hess_vx_vz, hess_vy_vz = 0, 0, 0
	
	hess_vx2 += np.dot(np.dot(invD, cprime_vx), cprime_vx)
	hess_vy2 += np.dot(np.dot(invD, cprime_vy), cprime_vy)
	hess_vz2 += np.dot(np.dot(invD, cprime_vz), cprime_vz)
	hess_pi2 += np.dot(np.dot(invD, cprime_pi), cprime_pi)
	hess_pi_vx += np.dot(np.dot(invD, cprime_pi), cprime_vx)
	hess_pi_vy += np.dot(np.dot(invD, cprime_pi), cprime_vy)
	hess_pi_vz += np.dot(np.dot(invD, cprime_pi), cprime_vz)
	hess_vx_vy += np.dot(np.dot(invD, cprime_vx), cprime_vy)
	hess_vx_vz += np.dot(np.dot(invD, cprime_vx), cprime_vz)
	hess_vy_vz += np.dot(np.dot(invD, cprime_vy), cprime_vz)
		

	hess_pi2 += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dpi)) + np.dot(invD,dD_dpi2))
	hess_pi2 += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dpi)), invD) - np.dot(np.dot(invD,dD_dpi2), invD))
				
	hess_sigmav2 += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dsigmav), np.dot(invD, dD_dsigmav)) + np.dot(invD,dD_dsigmav2))
	hess_sigmav2 += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dsigmav), np.dot(invD, dD_dsigmav)), invD) - np.dot(np.dot(invD, dD_dsigmav2), invD))
		
	hess_pi_sigmav += 0.5 * np.trace(-np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dsigmav)) + np.dot(invD,dD_dpisigmav))
	hess_pi_sigmav += 0.5 * np.tensordot(D, 2 * np.dot(np.dot(np.dot(invD, dD_dpi), np.dot(invD, dD_dsigmav)), invD) - np.dot(np.dot(invD, dD_dpisigmav), invD))
	
	hess[i,i] = hess_pi2
	hess[i,-1] = hess_pi_sigmav
	hess[i,-2] = hess_pi_vz 
	hess[i,-3] = hess_pi_vy
	hess[i,-4] = hess_pi_vx
	hess[-1,i] = hess[i,-1]
	hess[-2,i] = hess[i,-2]
	hess[-3,i] = hess[i,-3]
	hess[-4,i] = hess[i,-4]
		
	hess[-1,-1], hess[-2,-2], hess[-3,-3], hess[-4,-4] = hess_sigmav2, hess_vz2, hess_vy2, hess_vx2 
	hess[-2,-3], hess[-3,-4], hess[-2,-4] = hess_vy_vz, hess_vx_vy, hess_vx_vz
	hess[-3,-2], hess[-4,-3], hess[-4,-2] = hess[-2,-3], hess[-3,-4], hess[-2,-4]	
	
	return hess ### N = -E(H) for L

def stella_hess(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N):

	hess = stella_Nmatrix_full(init_par, alpha, delta, plx_obs, mualpha_obs, mudelta_obs, vrad_obs, sigma_obs, sigma_vrad, ccoef, N)
	return 2*hess ### -2 * -E(H) = 2*N
