#!/usr/bin/env python
# encoding: utf-8
"""
homography.py

Created by Yang Shuo on 2013-05-08.
Copyright (c) 2013 . All rights reserved.
"""

import sys
import os
from numpy import *

def normalize(points):    
	"""
		Normalize a collection of points
		in homogeneous coordinates so that
		last row = 1
	"""
	for row in points:
		row /= points[-1]
	return points

def make_homog(points):
	"""
		Convert a set of points (dim*n array)
		to homogeneous coordinates
	"""                          
	return vstack((points,ones((1,points.shape[1]))))

def H_from_points(fp,tp):
	"""
		Find homography H, such that fp is mapped
		to tp.
		Using the linear DLT method
		Points are conditioned automatically
	"""                                     
	if fp.shape != tp.shape:
		raise RuntimeError('points does not match')
	#condition points     
	
	# -- from points --
	m = mean(fp[:2],axis=1)
	maxstd = max(std(fp[:2],axis=1)) + 1e-9
	C1 = diag([1/maxstd,1/maxstd,1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp = dot(C1,fp)
	
	# -- to points --
	m = mean(tp[:2],axis=1)
	maxstd = max(std(tp[:2],axis=1)) + 1e-9
	C2 = diag([1/maxstd,1/maxstd,1])
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp = dot(C2,tp)
	
	#create matrix for linear method, 2 rows for eachh correspondence pair
	nbr_correspondences = fp.shape[1]
	A = zeros((2*nbr_correspondences,9))
	for i in range(nbr_correspondences):
		A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
				tp[0][i]*fp[0][i],tp[0][i]*fp[0][i],
				tp[0][i]]
		A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
				tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],
				tp[1][i]]		
	U,S,V = linalg.svd(A)
	H = V[8].reshape((3,3))
	# decondition
	H = dot(linalg.inv(C2),dot(H,C1))
	# normalize and return  
	return H/H[2,2]

def Haffine_from_points(fp,tp):
	"""
		Find H, affine transformation, such that fp is mapped
		to tp.
	"""                                     
	if fp.shape != tp.shape:
		raise RuntimeError('points does not match')
	#condition points     

	# -- from points --
	m = mean(fp[:2],axis=1)
	maxstd = max(std(fp[:2],axis=1)) + 1e-9
	C1 = diag([1/maxstd,1/maxstd,1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp_cond = dot(C1,fp)

	# -- to points --
	m = mean(tp[:2],axis=1)
	maxstd = max(std(tp[:2],axis=1)) + 1e-9
	C2 = C1.copy()
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp_cond = dot(C2,tp)

	# conditioned points have mean zero, so translation is zero
	A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
	U,S,V = linalg.svd(A.T)
	
	# create B and C matrices as Hartley-Zisserman (2:nd end) p 130
	tmp = V[:2].T
	B = tmp[:2]
	C = tmp[2:4]
	
	tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))),axis = 1)
	H = vstack((tmp2,[0,0,1]))
	# decondition
	H = dot(linalg.inv(C2),dot(H,C1))
	return H/H[2,2]  

def H_solve_s_and_ang(Mtx):
	if Mtx.shape != (2,2):
		raise RuntimeError('Input is not in right shape')
	   
	
def main():
	pass


if __name__ == '__main__':
	main()

