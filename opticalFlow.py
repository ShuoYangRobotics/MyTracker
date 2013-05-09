#!/usr/bin/env python
# encoding: utf-8
"""
opticalFlow.py

Created by Yang Shuo on 2013-05-07.
Copyright (c) 2013 . All rights reserved.

This code use SIFT + optical flow to do tracking.
It prerequisites a bounding box. 
First, SIFT of the entire image is extracted.
Second, optical flow of adjcent frames are calculated.
"""

import sys
import os     
import cv      
import cv2
from numpy import * 
from pylab import *
from PIL import Image
from time import *   
from homography import *
import sift

def mynorm(point1,point2):
	return sqrt((point1[0] - point2[0])*(point1[0] - point2[0])+(point1[1] - point2[1])*(point1[1] - point2[1]))

def normCrossCorrelation(img1, img2, points1, points2, status1): 
	similarity = zeros((len(points1),1))
	for i in xrange(len(points1)):
		if status1[i] == 1:         
			#print type(img1)
			patch1 = cv2.getRectSubPix(asarray(img1[:,:]),(10,10), points1[i])
			patch2 = cv2.getRectSubPix(asarray(img2[:,:]),(10,10), points2[i])
			result = cv2.matchTemplate(patch1,patch2,cv2.TM_CCOEFF_NORMED)
			#print result.tolist()[0][0]
			similarity[i] = result.tolist()[0][0]
			
		else:
			similarity[i] = 0.0 
			
	return similarity.tolist()

def filterPts(points1,points2,similarity,status,errors_fb):  
	med_sim = median(similarity)
	#print med_sim
	#print similarity 
	#print type(errors_fb)
	med_efb = median(errors_fb)
	
	len1 = len(points2)
	k = 0
	for i in xrange(len1):
		if similarity[i]>med_sim and status[i] == 1:
			points1[k] = points1[i]
			points2[k] = points2[i]
			k += 1
	
	# len1 = len(points2)
	# points2 = [points2[i] for i in xrange(len1) if errors_fb[i]<=med_efb and status[i] == 1]
	# points1 = [points1[i] for i in xrange(len1) if errors_fb[i]<=med_efb and status[i] == 1]
	#
	points1 = points1[:k]
	points2 = points2[:k]    
	if len(points1) == 0:
		print "zero!!" 
	return points1,points2

def windowWeight(point, ver1, ver2):
	"""
		2013-05-09
		This function takes in a point and two boundary points of a rectangle
		depends on the relative position of the point to the rectangle
		different weight is given  
	"""
	frac = 0.15                            
	wf = (ver2[0] - ver1[0])*frac
	hf = (ver2[1] - ver1[1])*frac
	if point[0]>= ver1[0]+wf and point[0]<= ver2[0]-wf:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return 1
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return (point[1] - ver1[1])/hf
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return (ver2[1] - point[1])/hf	
		else:			                 
			return 0.01
	elif point[0]>= ver1[0] and point[0]<= ver1[0]+wf:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return (point[0] - ver1[0])/wf
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return 0.02
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return 0.02	
		else:			                 
			return 0.01 		
	elif point[0]>= ver2[0]-wf and point[0]<= ver2[0]:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return (ver2[0]-point[0])/wf
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return 0.02
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return 0.02	
		else:
			return 0.01
	else:
		return 0.005 
                           			  
def main():                                                   
	path = "../rawData/outdoor1/GOPR0003_0/"
	files = os.listdir(path)  
	"""
		Get SIFT features for the first frame
	"""
	imname = path+files[10]
	a =  time()
	sift.process_image(imname,'empire.sift')          
	l1,d1 = sift.read_features_from_file('empire.sift')   
	b = time()
	print "Time for getting SIFT features: ",b - a
	refine_l = []
	#figure()
	#gray()   
	b_ver1 = (60,91)
	b_ver2 = (148,315) 
	for i,p in enumerate(l1):
		center = p[:2] 
		if center[0] > 60 and center[0] < 148 and center[1] > 91 and center[1] < 315:
			refine_l.append(tuple(center))
	#sift.plot_features(im1,l1,circle=True)
	#show()
	
	startIndex = 10
	writer = cv.CreateVideoWriter("myTrack.avi",cv.CV_FOURCC('M','J','P','G'),60,cv.GetSize (cv.LoadImage(path+files[startIndex])),1)
	
	while (startIndex<=2900):
		seq = [cv.LoadImage(path+files[startIndex]),
			cv.LoadImage(path+files[startIndex+1])]  
		#draw a box on image 1
		# cv.Rectangle(seq[0],(60,91),(148,315),(255,0,0))   
		# 	cv.ShowImage('TestOpticFlow', seq[0])   
		# 	
		# 	cv.WaitKey()
		# 	cv.DestroyAllWindows()                   
	
		"""
			calculate optical flow
		"""  
		color = (0,0,255)             
		newFrameImageGS_32F1 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		newFrameImageGS_32F2 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		cv.CvtColor(seq[0],newFrameImageGS_32F1,cv.CV_RGB2GRAY)
		cv.CvtColor(seq[1],newFrameImageGS_32F2,cv.CV_RGB2GRAY)
		pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		prev_pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		flags = 0 
		a =  time()     
	
		points, status1, errors,= cv.CalcOpticalFlowPyrLK (
		                    newFrameImageGS_32F1, 
		                    newFrameImageGS_32F2, 
		                    prev_pyramid, 
		                    pyramid,
		                    refine_l,
		                    (10,10),
		                    3,#pyr number
		                    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01),
		                    flags)
		
		"""center of mass of points"""
		p_x = [windowWeight(point,b_ver1,b_ver2)*point[0] for point in points]
		p_y = [windowWeight(point,b_ver1,b_ver2)*point[1] for point in points]		   
		weight = [windowWeight(point,b_ver1,b_ver2) for point in points]
		cm_x = sum(p_x)/sum(weight)
		cm_y = sum(p_y)/sum(weight)
		
		points = [point for point in points if mynorm(point,(cm_x,cm_y)) <= 0.7*mynorm(b_ver1,b_ver2)]
		
		old_cm = ((b_ver1[0]+b_ver2[0])/2,(b_ver1[1]+b_ver2[1])/2)
		b_ver1 = (int(b_ver1[0]+cm_x-old_cm[0]), int(b_ver1[1]+cm_y-old_cm[1]))
		b_ver2 = (int(b_ver2[0]+cm_x-old_cm[0]), int(b_ver2[1]+cm_y-old_cm[1]))		  	  	
		# points_fb, status2, errors_fb,= cv.CalcOpticalFlowPyrLK (
		#                     newFrameImageGS_32F2, 
		#                     newFrameImageGS_32F1, 
		#                     prev_pyramid, 
		#                     pyramid,
		#                     points,
		#                     (10,10),
		#                     3,#pyr number
		#                     (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01),
		#                     flags)
		# 
		# fb_error = [mynorm(points_fb[i],refine_l[i]) for i in xrange(len(points_fb))]
		# 
		# k = 0
		# for i,item in enumerate(fb_error):
		# 	if item < 0.01:
		# 		points[k] = points[i];
		#     	k += 1
		# 
		# points = points[:k]
		#     			
		
		# similarity = normCrossCorrelation(newFrameImageGS_32F1, 
		# 						newFrameImageGS_32F2, 
		# 						refine_l, 
		# 						points,
		# 						status1) 
	   	#refine_l,points = filterPts(refine_l,points,similarity,status1,fb_error)
		
		b = time()
		print "Time for calculating optical flow: ",b - a
	   
		#a =  time()
		#fp = make_homog(transpose(array(refine_l)))
		#tp = make_homog(transpose(array(points)))
		#H = H_from_points(fp,tp) 
		#H = Haffine_from_points(fp,tp)
		#b = time()
		#print "Time for calculating homography: ",b - a  
		#print H[0:2,0:2]  
		#H_solve_s_and_ang(H[0:2,0:2])
		#print H[0:2,2]
		#disp = (int(H[0:2,2][0]),int(H[0:2,2][1]))
	
		
		for pt in refine_l:
			cv.Circle(seq[0], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.Circle(seq[0], (int(cm_x), int(cm_y)), 10, (0,255,255), 0, cv.CV_AA, 0)	 
		cv.Rectangle(seq[0],b_ver1,b_ver2,(255,0,0))
		cv.ShowImage('First', seq[0])   
		cv.WaitKey(30)
		
		#b_ver1 = (b_ver1[0]+disp[0],b_ver1[1]+disp[1])
		#b_ver2 = (b_ver2[0]+disp[0],b_ver2[1]+disp[1])
		
		cv.Rectangle(seq[1],b_ver1,b_ver2,(255,0,0)) 
		#cv.DestroyWindow('First')
		for pt in points:
			cv.Circle(seq[1], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.Circle(seq[1], (int(cm_x), int(cm_y)), 10, (0,255,255), 0, cv.CV_AA, 0)	
		cv.Rectangle(seq[1],b_ver1,b_ver2,(255,0,0))
		cv.ShowImage('First', seq[1])
		#sleep(0.7)  
		cv.WaitKey(5)
		#cv.DestroyWindow('First')    
		
		refine_l = points
		startIndex += 1
		cv.WriteFrame(writer, seq[0])
		cv.WriteFrame(writer, seq[1])

if __name__ == '__main__':
	main()

