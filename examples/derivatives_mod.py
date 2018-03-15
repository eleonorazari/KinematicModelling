### Purpose: compute the gradient (grad) and the hessian (Nmatrix, hess) of the likelihood function. 
### Note: these functions work but they  could/should be written better to 
### avoid repetitions and to increase the code speed.
### The mathematical formulae implemented here can be found in the appendix of
### Lindegren et al. (2000).

import numpy as np
from pygaia.astrometry.vectorastrometry import phaseSpaceToAstrometry, normalTriad
from pygaia.astrometry.constants import *
from matrixOperations import *


_A = auKmYearPerSec  ### km*yr/s


def gradient(init_par, alpha, delta, obs, sigma_obs, ccoef, N):
	"""
	Estimate gradient of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				       2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       3) The cluster velocity dispersion, sigma_v [km/s];
	alpha, delta - Cluster member positions [rad];
	obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	Returns:
	-----------
	f - An array with n+4 elements corresponding to the gradient of the likelihood 
		   with respect to the n+4 variables.
	"""


	## Initial parameters

	parallax, v, sigma_v = init_par[:-4], init_par[-4:-1], init_par[-1] 
	plx_obs, mualpha_obs, mudelta_obs = obs[:, 0], obs[:, 1], obs[:, 2]

	### Define normal triad and proper motions
	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A
	
	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)
	a,like, expo, detD = np.ones(N),np.ones(N),np.ones(N), np.ones(N) 

	### Eq. 8 in Lindegren+2000 (Covariance Matrix)
	C = np.zeros((3,3,N),dtype=np.float64)
	C[0,0,:],C[1,1,:],C[2,2,:] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta = np.zeros(N), np.zeros(N), np.zeros(N)
	corr_coefficient_plx_mualpha[:], corr_coefficient_plx_mudelta[:], corr_coefficient_mualpha_mudelta[:] = ccoef[:, 0], ccoef[:, 1], ccoef[:, 2] 
	
	C[0,1,:], C[0,2,:] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0,:], C[1,2,:] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0,:], C[2,1,:] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta

	### Eq. 16 in Lindegren+2000 (Definition of D matrix)	
	E = np.zeros((3,3,N),dtype=np.float64)
	E[1,1,:],E[2,2,:] = (sigma_v*parallax[:]/_A)**2., (sigma_v*parallax[:]/_A)**2.
	D,invD = np.zeros((3,3,N),dtype=np.float64),np.zeros((3,3,N),dtype=np.float64)
	D = np.add(E,C)
	for i in range(N):
		detD[i] =  matrix_det(D[:,:,i]) 
		invD[:,:,i] =  matrix_inv(D[:,:,i])
		
	
	a_c = np.ones((3,N))
	a_c = [plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs-mudelta_mod]
	
	### First derivatives in Eq. A3 
	cprime_pi, cprime_vx, cprime_vy, cprime_vz,  = np.ones((3,N)), np.ones((3,N)), \
							np.ones((3,N)), np.ones((3,N)), 
	cprime_pi[0,:] = 1.
	cprime_pi[1,:] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2,:] = np.dot(np.transpose(q),v)/_A
	
	cprime_vx[0,:] = 0.
	cprime_vx[1,:] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2,:] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A

	
	cprime_vy[0,:] = 0.
	cprime_vy[1,:] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2,:] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A

	cprime_vz[0,:] = 0.
	cprime_vz[1,:] = 0. 
	cprime_vz[2,:] = np.cos(delta)*plx_mod/_A

	dlnd_dpi, dlnd_dsigmav  = np.zeros(N),  np.zeros(N)
	de_dpi, de_dsigmav  = np.zeros(N),  np.zeros(N)
	

	### See Eq. A5 
	de_dpi[:] = ((sigma_v/_A)**2.)*2.*plx_mod[:]
	de_dsigmav[:] = ((plx_mod[:]/_A)**2.)*2.*sigma_v
	
	dlnd_dpi[:] = (invD[1,1,:] + invD[2,2,:])*de_dpi[:]  
	dlnd_dsigmav[:] = (invD[1,1,:] + invD[2,2,:])*de_dsigmav[:]
	
	
	
	### See Eq. A6
	dG_dpi, dG_dsigmav = np.zeros((3,3,N)), np.zeros((3,3,N)) 
	
	dG_dpi[0,0,:], dG_dpi[0,1,:], dG_dpi[0,2,:] = (-invD[0,1,:]*invD[1, 0, :] - invD[0, 2, :]*invD[2,0,:])*de_dpi[:], \
						      (-invD[0,1,:]*invD[1, 1, :] - invD[0,2,:]*invD[2, 1, :])*de_dpi[:], \
						      (-invD[0,1,:]*invD[1,2,:] - invD[0,2,:]*invD[2,2,:])*de_dpi[:]
	dG_dpi[1,0,:], dG_dpi[1,1,:], dG_dpi[1,2,:] = (-invD[1,1,:]*invD[1, 0, :] - invD[1, 2, :]*invD[2,0,:])*de_dpi[:], \
						      (-invD[1,1,:]*invD[1, 1, :] - invD[1,2,:]*invD[2, 1, :])*de_dpi[:], \
						      (-invD[1,1,:]*invD[1,2,:] - invD[1,2,:]*invD[2,2,:])*de_dpi[:]
	dG_dpi[2,0,:], dG_dpi[2,1,:], dG_dpi[2,2,:] = (-invD[2,1,:]*invD[1, 0, :] - invD[2, 2, :]*invD[2,0,:])*de_dpi[:], \
						      (-invD[2,1,:]*invD[1, 1, :] - invD[2,2,:]*invD[2, 1, :])*de_dpi[:], \
						      (-invD[2,1,:]*invD[1,2,:] - invD[2,2,:]*invD[2,2,:])*de_dpi[:]
	

	dG_dsigmav[0,0,:], dG_dsigmav[0,1,:], dG_dsigmav[0,2,:] = (-invD[0,1,:]*invD[1, 0, :] - invD[0, 2, :]*invD[2,0,:])*de_dsigmav[:], \
								  (-invD[0,1,:]*invD[1, 1, :] - invD[0,2,:]*invD[2, 1, :])*de_dsigmav[:], \
								  (-invD[0,1,:]*invD[1,2,:] - invD[0,2,:]*invD[2,2,:])*de_dsigmav[:]
	dG_dsigmav[1,0,:], dG_dsigmav[1,1,:], dG_dsigmav[1,2,:] = (-invD[1,1,:]*invD[1, 0, :] - invD[1, 2, :]*invD[2,0,:])*de_dsigmav[:], \
								  (-invD[1,1,:]*invD[1, 1, :] - invD[1,2,:]*invD[2, 1, :])*de_dsigmav[:], \
								  (-invD[1,1,:]*invD[1,2,:] - invD[1,2,:]*invD[2,2,:])*de_dsigmav[:]
	dG_dsigmav[2,0,:], dG_dsigmav[2,1,:], dG_dsigmav[2,2,:] = (-invD[2,1,:]*invD[1, 0, :] - invD[2, 2, :]*invD[2,0,:])*de_dsigmav[:], \
								  (-invD[2,1,:]*invD[1, 1, :] - invD[2,2,:]*invD[2, 1, :])*de_dsigmav[:], \
								  (-invD[2,1,:]*invD[1,2,:] - invD[2,2,:]*invD[2,2,:])*de_dsigmav[:]

	f_dpi = np.zeros((N), dtype=np.float64) 
	
	
	for i in range(N):
		f_dpi_1, f_dpi_3 = 0., 0.0 
		for ia in range(3):
			for ib in range(3):
				f_dpi_1 += invD[ia,ib,i]*cprime_pi[ia,i]*a_c[ib][i]
				f_dpi_3 += (-0.5)*(dG_dpi[ia,ib,i]*a_c[ia][i]*a_c[ib][i])
					
		f_dpi_2 = (-0.5)*dlnd_dpi[i]
		f_dpi[i] = f_dpi_1 + f_dpi_2 + f_dpi_3
		

	f_vx, f_vy, f_vz, f_sigmav = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N) 

	f_vx = np.sum(invD[0,0,:]*cprime_vx[0,:]*a_c[0][:] + invD[0,1,:]*cprime_vx[0,:]*a_c[1][:] + invD[0,2,:]*cprime_vx[0,:]*a_c[2][:] + \
		   invD[1,0,:]*cprime_vx[1,:]*a_c[0][:] + invD[1,1,:]*cprime_vx[1,:]*a_c[1][:] + invD[1,2,:]*cprime_vx[1,:]*a_c[2][:] + \
		   invD[2,0,:]*cprime_vx[2,:]*a_c[0][:] + invD[2,1,:]*cprime_vx[2,:]*a_c[1][:] + invD[2,2,:]*cprime_vx[2,:]*a_c[2][:])
	
	f_vy = np.sum(invD[0,0,:]*cprime_vy[0,:]*a_c[0][:] + invD[0,1,:]*cprime_vy[0,:]*a_c[1][:] + invD[0,2,:]*cprime_vy[0,:]*a_c[2][:] + \
		   invD[1,0,:]*cprime_vy[1,:]*a_c[0][:] + invD[1,1,:]*cprime_vy[1,:]*a_c[1][:] + invD[1,2,:]*cprime_vy[1][:]*a_c[2][:] + \
		   invD[2,0,:]*cprime_vy[2,:]*a_c[0][:] + invD[2,1,:]*cprime_vy[2,:]*a_c[1][:] + invD[2,2,:]*cprime_vy[2,:]*a_c[2][:])

	f_vz = np.sum(invD[0,0,:]*cprime_vz[0,:]*a_c[0][:] + invD[0,1,:]*cprime_vz[0,:]*a_c[1][:] + invD[0,2,:]*cprime_vz[0,:]*a_c[2][:] + \
		   invD[1,0,:]*cprime_vz[1,:]*a_c[0][:] + invD[1,1,:]*cprime_vz[1,:]*a_c[1][:] + invD[1,2,:]*cprime_vz[1,:]*a_c[2][:] + \
		   invD[2,0,:]*cprime_vz[2,:]*a_c[0][:] + invD[2,1,:]*cprime_vz[2,:]*a_c[1][:] + invD[2,2,:]*cprime_vz[2,:]*a_c[2][:])
	
	f_sigmav = np.sum(-0.5*(dG_dsigmav[0,0,:]*a_c[0][:]*a_c[0][:] + dG_dsigmav[0,1,:]*a_c[1][:]*a_c[0][:]+ dG_dsigmav[0,2,:]*a_c[2][:]*a_c[0][:] + \
		    dG_dsigmav[1,0,i]*a_c[1][:]*a_c[0][:] + dG_dsigmav[1,1,:]*a_c[1][:]*a_c[1][:]+ dG_dsigmav[1,2,:]*a_c[1][:]*a_c[2][:] + 	
		    dG_dsigmav[2,0,i]*a_c[2][:]*a_c[0][:] + dG_dsigmav[2,1,:]*a_c[2][:]*a_c[1][:]+ dG_dsigmav[2,2,:]*a_c[2][:]*a_c[2][:]))
	

	f_sigmav = f_sigmav - 0.5*np.sum(dlnd_dsigmav)	
	f = np.concatenate((f_dpi, np.array([f_vx, f_vy, f_vz, f_sigmav]))) ### Grad L(theta), see Eq. 17
	return -2.*f   						    ### Grad U(theta), see Eq. 18


def Nmatrix(init_par, alpha, delta, obs, sigma_obs, ccoef, N):
	"""
	Estimate covariance matrix of the likelihood function.

	Parameters:
	------------
	init_par - Set of initial values for: 1) All the parallaxes [mas];
				     	      2) The cluster centroid velocity  [vx_0, vy_0, vz_0] [km/s];
				       	      3) The cluster velocity dispersion [km/s];
	alpha, delta - Cluster member positions [rad];
	obs - observed values for parallaxes and proper motions [mas, mas/yr];
	sigma_obs - observed errors for parallaxes and proper motions [mas, mas/yr];
	ccoef - 3-dim array of correlation coefficients from the HIP catalogue;
	N - number of stars;
	Returns:
	-----------
	Cov - N+4 x N+4 matrix, corresponding to the covariance of the parameters of the likelihood function.
	"""
	parallax, v, sigma_v = init_par[:-4], init_par[-4:-1], init_par[-1] 
	plx_obs, mualpha_obs, mudelta_obs = obs[:, 0], obs[:, 1], obs[:, 2]

	p, q, r = normalTriad(alpha, delta)
	mualpha_mod = np.dot(np.transpose(p),v)*parallax/_A
	mudelta_mod = np.dot(np.transpose(q),v)*parallax/_A
	
	plx_mod, mualpha_mod, mudelta_mod = parallax, mualpha_mod, mudelta_mod
	sigma_plx, sigma_mualpha, sigma_mudelta = np.transpose(sigma_obs)
	a,like, expo, detD = np.ones(N),np.ones(N),np.ones(N), np.ones(N) 
	C = np.zeros((3,3,N),dtype=np.float64)
	C[0,0,:],C[1,1,:],C[2,2,:] = sigma_plx**2.,sigma_mualpha**2., sigma_mudelta**2.
	corr_coefficient_plx_mualpha, corr_coefficient_plx_mudelta, corr_coefficient_mualpha_mudelta = np.zeros(N), np.zeros(N), np.zeros(N)
	corr_coefficient_plx_mualpha[:], corr_coefficient_plx_mudelta[:], corr_coefficient_mualpha_mudelta[:] = ccoef[:, 0], ccoef[:, 1], ccoef[:, 2] 
	
	C[0,1,:], C[0,2,:] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta
	C[1,0,:], C[1,2,:] = corr_coefficient_plx_mualpha*sigma_plx*sigma_mualpha, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	C[2,0,:], C[2,1,:] = corr_coefficient_plx_mudelta*sigma_plx*sigma_mudelta, corr_coefficient_mualpha_mudelta*sigma_mualpha*sigma_mudelta
	E = np.zeros((3,3,N),dtype=np.float64)
	E[1,1,:],E[2,2,:] = (sigma_v**2.)*(parallax/_A)**2., (sigma_v**2.)*(parallax/_A)**2.
	D,invD = np.zeros((3,3,N),dtype=np.float64),np.zeros((3,3,N),dtype=np.float64)
	D = np.add(E,C)
	for i in range(N):
		detD[i] =  matrix_det(D[:,:,i]) 
		invD[:,:,i] =  matrix_inv(D[:,:,i])
		
	a_c = np.ones((3,N))
	a_c = [plx_obs - plx_mod, mualpha_obs - mualpha_mod, mudelta_obs-mudelta_mod]
	
	

	
	cprime_pi, cprime_vx, cprime_vy, cprime_vz,  = np.ones((3,N)), np.ones((3,N)), \
							np.ones((3,N)), np.ones((3,N)), 
	cprime_pi[0,:] = 1.
	cprime_pi[1,:] = np.dot(np.transpose(p),v)/_A
	cprime_pi[2,:] = np.dot(np.transpose(q),v)/_A
	
	cprime_vx[0,:] = 0.
	cprime_vx[1,:] = -np.sin(alpha)*plx_mod/_A 
	cprime_vx[2,:] = -np.sin(delta)*np.cos(alpha)*plx_mod/_A

	
	cprime_vy[0,:] = 0.
	cprime_vy[1,:] = np.cos(alpha)*plx_mod/_A 
	cprime_vy[2,:] = -np.sin(delta)*np.sin(alpha)*plx_mod/_A

	cprime_vz[0,:] = 0.
	cprime_vz[1,:] = 0. 
	cprime_vz[2,:] = np.cos(delta)*plx_mod/_A

	dlnd_dpi, dlnd_dsigmav  = np.zeros(N),  np.zeros(N)
	de_dpi, de_dsigmav  = np.zeros(N),  np.zeros(N)
	

	### See formula A.5 
	de_dpi[:] = ((sigma_v/_A)**2.)*2.*plx_mod[:]
	de_dsigmav[:] = ((plx_mod[:]/_A)**2.)*2.*sigma_v
	
	dlnd_dpi[:] = (invD[1,1,:] + invD[2,2,:])*de_dpi[:]  
	dlnd_dsigmav[:] = (invD[1,1,:] + invD[2,2,:])*de_dsigmav[:]
	
	
	
	### See formula A.7
	hess = np.zeros((N+4, N+4))

	hess_diag_pi, hess_diag_pi_1, hess_diag_pi_2  = np.zeros(N), np.zeros(N), np.zeros(N)
	hess_diag_pi_1[:] = invD[0, 0, :]*cprime_pi[0, :]*cprime_pi[0, :] + invD[0, 1, :]*cprime_pi[0, :]*cprime_pi[1, :] + invD[0, 2, :]*cprime_pi[0, :]*cprime_pi[2, :] + \
			    invD[1, 0, :]*cprime_pi[1, :]*cprime_pi[0, :] + invD[1, 1, :]*cprime_pi[1, :]*cprime_pi[1, :] + invD[1, 2, :]*cprime_pi[1, :]*cprime_pi[2, :] + \
		 	    invD[2, 0, :]*cprime_pi[2, :]*cprime_pi[0, :] + invD[2, 1, :]*cprime_pi[2, :]*cprime_pi[1, :] + invD[2, 2, :]*cprime_pi[2, :]*cprime_pi[2, :]	
	
	
	#hess_diag_pi_2[:] = np.sum(0.5*(invD[1, 1, :]**2. + 2.*invD[1, 2, :]**2. + invD[2, 2, :]**2.)*de_dpi[:]*de_dpi[:]) ### Check if it's with or without sum: without!
	# So correct formula is below.
	hess_diag_pi_2[:] = (0.5*(invD[1, 1, :]**2. + 2.*invD[1, 2, :]**2. + invD[2, 2, :]**2.)*de_dpi[:]*de_dpi[:])
	hess_diag_pi[:] = hess_diag_pi_1[:] + hess_diag_pi_2[:]	

	
	hess_diag_vx, hess_diag_vy, hess_diag_vz, hess_diag_sigmav = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
	hess_pi_vx, hess_pi_vy, hess_pi_vz, hess_pi_sigmav = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
	hess_diag_vxi, hess_diag_vyi, hess_diag_vzi = np.zeros(N), np.zeros(N), np.zeros(N)
	
	hess_diag_vxi[:] = invD[0, 0, :]*cprime_vx[0, :]*cprime_vx[0, :] + invD[0, 1, :]*cprime_vx[0, :]*cprime_vx[1, :] + invD[0, 2, :]*cprime_vx[0, :]*cprime_vx[2, :] + \
			   invD[1, 0, :]*cprime_vx[1, :]*cprime_vx[0, :] + invD[1, 1, :]*cprime_vx[1, :]*cprime_vx[1, :] + invD[1, 2, :]*cprime_vx[1, :]*cprime_vx[2, :] + \
			   invD[2, 0, :]*cprime_vx[2, :]*cprime_vx[0, :] + invD[2, 1, :]*cprime_vx[2, :]*cprime_vx[1, :] + invD[2, 2, :]*cprime_vx[2, :]*cprime_vx[2, :] 		
	
	hess_diag_vyi[:] = invD[0, 0, :]*cprime_vy[0, :]*cprime_vy[0, :] + invD[0, 1, :]*cprime_vy[0, :]*cprime_vy[1, :] + invD[0, 2, :]*cprime_vy[0, :]*cprime_vy[2, :] +\
			   invD[1, 0, :]*cprime_vy[1, :]*cprime_vy[0, :] + invD[1, 1, :]*cprime_vy[1, :]*cprime_vy[1, :] + invD[1, 2, :]*cprime_vy[1, :]*cprime_vy[2, :] +\
			   invD[2, 0, :]*cprime_vy[2, :]*cprime_vy[0, :] + invD[2, 1, :]*cprime_vy[2, :]*cprime_vy[1, :] + invD[2, 2, :]*cprime_vy[2, :]*cprime_vy[2, :] 	


	hess_diag_vzi[:] = invD[0, 0, :]*cprime_vz[0, :]*cprime_vz[0, :] + invD[0, 1, :]*cprime_vz[0, :]*cprime_vz[1, :] + invD[0, 2, :]*cprime_vz[0, :]*cprime_vz[2, :] +\
			   invD[1, 0, :]*cprime_vz[1, :]*cprime_vz[0, :] + invD[1, 1, :]*cprime_vz[1, :]*cprime_vz[1, :] + invD[1, 2, :]*cprime_vz[1, :]*cprime_vz[2, :] +\
			   invD[2, 0, :]*cprime_vz[2, :]*cprime_vz[0, :] + invD[2, 1, :]*cprime_vz[2, :]*cprime_vz[1, :] + invD[2, 2, :]*cprime_vz[2, :]*cprime_vz[2, :] 		
	

	hess_pi_vx[:] = invD[0, 0, :]*cprime_pi[0,:]*cprime_vx[0, :] + invD[0, 1, :]*cprime_pi[0,:]*cprime_vx[1, :] + invD[0, 2, :]*cprime_pi[0,:]*cprime_vx[2, :] +\
			invD[1, 0, :]*cprime_pi[1,:]*cprime_vx[0, :] + invD[1, 1, :]*cprime_pi[1,:]*cprime_vx[1, :] + invD[1, 2, :]*cprime_pi[1,:]*cprime_vx[2, :] +\
			invD[2, 0, :]*cprime_pi[2,:]*cprime_vx[0, :] + invD[2, 1, :]*cprime_pi[2,:]*cprime_vx[1, :] + invD[2, 2, :]*cprime_pi[2,:]*cprime_vx[2, :] 

	hess_pi_vy[:] = invD[0, 0, :]*cprime_pi[0,:]*cprime_vy[0, :] + invD[0, 1, :]*cprime_pi[0,:]*cprime_vy[1, :] + invD[0, 2, :]*cprime_pi[0,:]*cprime_vy[2, :] +\
			invD[1, 0, :]*cprime_pi[1,:]*cprime_vy[0, :] + invD[1, 1, :]*cprime_pi[1,:]*cprime_vy[1, :] + invD[1, 2, :]*cprime_pi[1,:]*cprime_vy[2, :] +\
			invD[2, 0, :]*cprime_pi[2,:]*cprime_vy[0, :] + invD[2, 1, :]*cprime_pi[2,:]*cprime_vy[1, :] + invD[2, 2, :]*cprime_pi[2,:]*cprime_vy[2, :] 

	hess_pi_vz[:] = invD[0, 0, :]*cprime_pi[0,:]*cprime_vz[0, :] + invD[0, 1, :]*cprime_pi[0,:]*cprime_vz[1, :] + invD[0, 2, :]*cprime_pi[0,:]*cprime_vz[2, :] +\
			invD[1, 0, :]*cprime_pi[1,:]*cprime_vz[0, :] + invD[1, 1, :]*cprime_pi[1,:]*cprime_vz[1, :] + invD[1, 2, :]*cprime_pi[1,:]*cprime_vz[2, :] +\
			invD[2, 0, :]*cprime_pi[2,:]*cprime_vz[0, :] + invD[2, 1, :]*cprime_pi[2,:]*cprime_vz[1, :] + invD[2, 2, :]*cprime_pi[2,:]*cprime_vz[2, :] 

						
	hess_diag_vx = np.sum(hess_diag_vxi)
	hess_diag_vy = np.sum(hess_diag_vyi)
	hess_diag_vz = np.sum(hess_diag_vzi)	
	
	hess_diag_sigmav = np.sum(0.5*(invD[1, 1, :]**2. + 2.*invD[1, 2, :]**2. + invD[2, 2, :]**2.)*de_dsigmav[:]*de_dsigmav[:])
	hess_pi_sigmav[:] = 0.5*(invD[1, 1, :]**2. + 2.*invD[1, 2, :]**2. + invD[2, 2, :]**2.)*de_dpi[:]*de_dsigmav[:] 

	hess_diag = np.concatenate((hess_diag_pi, np.array([hess_diag_vx, hess_diag_vy, hess_diag_vz, hess_diag_sigmav])))
	
	for i in range(N+4):
		hess[i, i] = hess_diag[i]
		
	
	for j in range(N):
			hess[j, -4] = hess_pi_vx[j]
			hess[j, -3] = hess_pi_vy[j]
			hess[j, -2] = hess_pi_vz[j]
			hess[j, -1] = hess_pi_sigmav[j]
			hess[-4, j] = hess_pi_vx[j]
			hess[-3, j] = hess_pi_vy[j] 
			hess[-2, j] = hess_pi_vz[j]
			hess[-1, j] = hess_pi_sigmav[j]
			

	
	
	part_12, part_13, part_23  = np.zeros(N),np.zeros(N),np.zeros(N)
	for ia in range(3):
		for ib in range(3):
			part_12[:] += invD[ia, ib, :]*cprime_vx[ia, :]*cprime_vy[ib, :] 
			part_13[:] += invD[ia, ib, :]*cprime_vx[ia, :]*cprime_vz[ib, :] 
			part_23[:] += invD[ia, ib, :]*cprime_vy[ia, :]*cprime_vz[ib, :] 				

	hess[-4, -3] = np.sum(part_12)
	hess[-3, -4] = hess[-4, -3]
	
	hess[-4, -2] = np.sum(part_13)
	hess[-2, -4] = hess[-4, -2]

	hess[-3, -2] = np.sum(part_23)
	hess[-2, -3] = hess[-3, -2]

	#### I am returning here the matrix Njk, which is defined as -E(H),
	#### where H is the hessian of the likelihood: therefore to obtain the real hessian, one
	#### should multiply this by '-1' (see function below.)
	return hess ### See eq. 18



def hessian(init_par, alpha, delta, obs,  sigma_obs, ccoef, N):

	#### In this function, I return the hessian of the function U, which is what I am minimizing
	#### When I want to estimate the errors on the quantities, I should actually use the function above,
	### remembering that I have to multiply first by '-1', invert, and then multiply again by '-1' 

	hessL = Nmatrix(init_par, alpha, delta, obs,  sigma_obs, ccoef, N)
	hessL = -1.*hessL
	hessU = -2.*hessL
	return hessU
 

