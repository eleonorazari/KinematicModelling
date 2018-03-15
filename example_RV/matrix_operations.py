# Purpose: some useful functions to deal with matrixes.
# These routines are already implemented in python (linalg)
# But here they are faster because there are less controls 
# (especially in matrix_inv it could probably be better to 
# add the option of determinant = 0).


import numpy as np



def matrix_det(A):

	"""
	Returns the determinant of A, a 3x3 matrix.
	You can use also scipy.linalg.det.
	Might check which is faster.

	Parameters
   	----------

    	A - input 3x3 matrix

	"""
	x = A[0,0]*A[1,1]*A[2,2] + A[0,1]*A[1,2]*A[2,0] + A[0,2]*A[1,0]*A[2,1]
	y = A[0,0]*A[1,2]*A[2,1] + A[0,1]*A[1,0]*A[2,2] + A[0,2]*A[1,1]*A[2,0]
	return x - y


def matrix_inv(mat):
	"""
	Calculate the inverse of a 3x3 matrix. This code is specialized for the 3x3 case and
	hopefully faster than standard python functions.

	NOTE that no precautions are taken to ensure numerically stable calculations. The code is just a
	straightforward implementation of the formal mathematical inverse of a 3x3 matrix.

	Parameters
	----------

	mat - input 3x3 matrix

	Returns
	-------

	invmat - inverse matrix
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
	return invmat

def row_matrix_col(a, b, A):

	"""
	Compute the product a'Ab.
	Parameters:
	------------
	a - 3 elements array (a' is its transpose)
	b - 3 elements array
	A - 3x3 matrix 

	Returns:
	------------
	The value of the product.

	Notes:
	------------
	This could be generalized to higher dimensions.

	Example:
	(a b c)| A[0,0] A[0,1] A[0,2] | (x y z)' =  
	       | A[1,0] A[1,1] A[1,2] |	
	       | A[2,0] A[2,1] A[2,2] |
	
	= aA[0,0]x + bA[1,0]x + cA[2,0]x 
	  + aA[0,1]y + bA[1,1]y + cA[2,1]y + 
	  + aA[0,2]z + bA[1,2]z + cA[2,2]z
	"""

	return (a[0]*A[0][0]*b[0] + a[1]*A[1][0]*b[0] + a[2]*A[2][0]*b[0] + 
	       a[0]*A[0][1]*b[1] + a[1]*A[1][1]*b[1] + a[2]*A[2][1]*b[1] +  
	       a[0]*A[0][2]*b[2] + a[1]*A[1][2]*b[2] + a[2]*A[2][2]*b[2])


def sum_matrixes(A, B):

	"""
	Perform the sum of the elements of 2 3x3 matrixes.

	Parameters:
	------------
	A - 3x3 matrix
	B - 3x3 matrix
	
	Returns:
	D - 3x3 matrix

	"""

	D = np.zeros((3,3))
	D[0][0] = A[0][0]+ B[0][0]
	D[0][1] = A[0][1]+ B[0][1]
	D[0][2] = A[0][2]+ B[0][2]
	D[1][0] = A[1][0]+ B[1][0]
	D[1][1] = A[1][1]+ B[1][1]
	D[1][2] = A[1][2]+ B[1][2]
	D[2][0] = A[2][0]+ B[2][0]
	D[2][1] = A[2][1]+ B[2][1]
	D[2][2] = A[2][2]+ B[2][2]
	return D


def row_matrix_col_4d(a, b, A):

	"""
	Compute the product a'Ab.
	Parameters:
	------------
	a - 4 elements array (a' is its transpose)
	b - 4 elements array
	A - 4x4 matrix 

	Returns:
	------------
	The value of the product.

	Notes:
	------------
	This could be generalized to higher dimensions.

	Example:
	(a b c d)| A[0,0] A[0,1] A[0,2]  A[0,3]| (x y z w)' =  
	       	 | A[1,0] A[1,1] A[1,2]  A[1,3]|	
	       	 | A[2,0] A[2,1] A[2,2]  A[2,3]|
		 | A[3,0] A[3,1] A[3,2]  A[3,3]|
	
	= (aA[0, 0] + bA[1,0] + cA[2,0] + dA[3,0],
	   aA[0,1] + bA[1,1] + cA[2,1] + dA[3,1],
	   aA[0,2] + bA[1,2] + cA[2,2] + dA[3,2], 
	   aA[0,3] + bA[1,3] + cA[2,3] + dA[3,3]) (x y z w)'

	=   aA[0,0]x + bA[1,0]x + cA[2,0]x + dA[3,0]x +
	    aA[0,1]y + bA[1,1]y + cA[2,1]y + dA[3,1]y +	
	    aA[0,2]z + bA[1,2]z + cA[2,2]z + dA[3,2]z +
	    aA[0,3]w + bA[1,3]w + cA[2,3]w + dA[3,3]w	

	"""

	return (a[0]*A[0][0]*b[0] + a[1]*A[1][0]*b[0] + a[2]*A[2][0]*b[0] + a[3]*A[3][0]*b[0] +
	       a[0]*A[0][1]*b[1] + a[1]*A[1][1]*b[1] + a[2]*A[2][1]*b[1] +  a[3]*A[3][1]*b[1] +
	       a[0]*A[0][2]*b[2] + a[1]*A[1][2]*b[2] + a[2]*A[2][2]*b[2] +  a[3]*A[3][2]*b[2] +
	       a[0]*A[0][3]*b[3] + a[1]*A[1][3]*b[3] + a[2]*A[2][3]*b[3] +  a[3]*A[3][3]*b[3])



