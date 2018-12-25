
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import random



# In[4]:


def SIFTKeyP(Grayimage,colorImage):
    image_copy = colorImage.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints,decriptor = sift.detectAndCompute(Grayimage,None)
    keyPointImage=cv2.drawKeypoints(Grayimage,keyPoints,image_copy)
    return keyPointImage,keyPoints,decriptor

def matchingKeypoints(keyimage1,keyimage2,desc1,desc2,image1,image2):
    image1_copy = image1.copy()
    image2_copy = image2.copy()
    #using a brute force matcher
    bruteForce = cv2.BFMatcher()
    matches = bruteForce.knnMatch(desc1,desc2, k=2)

    #finding the best match out of the two
    goodMatches = []
    for (x,y) in matches:
        if x.distance < 0.75*y.distance:
            #matchesMask[i]=[1,0]
            goodMatches.append(x)

    outImage = None
    image = cv2.drawMatches(image1_copy,keyimage1,image2_copy,keyimage2,goodMatches,outImage,flags=2)
    return image,goodMatches

def fundamentalMatrix(keyPoint1,keyPoint2,goodMatches,imageM1,imageM2):
    src_pts = np.float32([ keyPoint1[m.queryIdx].pt for m in goodMatches ])
    dst_pts = np.float32([ keyPoint2[m.trainIdx].pt for m in goodMatches ])

    FMatrix, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)
    #print(FMatrix)
    
    src_pts = src_pts[mask.ravel()==1]
    dst_pts = dst_pts[mask.ravel()==1]
    
    
    leftEpilines = cv2.computeCorrespondEpilines(dst_pts[10:20].reshape(-1,1,2),2,FMatrix)
    leftEpilines = leftEpilines.reshape(-1,3)
    epiImageLeft = drawEpiLines(imageM1,imageM2,leftEpilines,src_pts[10:20],dst_pts[10:20])
    
    rightEpiLines = cv2.computeCorrespondEpilines(src_pts[10:20].reshape(-1,1,2), 1,FMatrix)
    rightEpiLines = rightEpiLines.reshape(-1,3)
    epiImageRight = drawEpiLines(imageM2,imageM1,rightEpiLines,src_pts[10:20],dst_pts[10:20])
    
    return FMatrix,epiImageLeft,epiImageRight

def drawEpiLines(lineOnImge,correspondingImage,lines,points1,points2):
    r,c = lineOnImge.shape
    #print(len(lines))
    lineOnImge = cv2.cvtColor(lineOnImge,cv2.COLOR_GRAY2BGR)
    correspondingImage = cv2.cvtColor(correspondingImage,cv2.COLOR_GRAY2BGR)
    color = [(227, 101, 113),(255,255,255),(128,0,0),
(105, 6, 91),(63, 254, 249),(255,69,0),(230, 11, 48),(190, 20, 223),(123, 189, 187),(255, 205, 30)]
    i = 0
    #print(len(lines))
    for r,p1,p2 in zip(lines,points2,points2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        lineOnImge = cv2.line(lineOnImge, (x0,y0), (x1,y1), color[i],1)
        i = i+1
    return lineOnImge

def disparity(image1,image2):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=17)
    #disparity = stereo.compute(image1,image2).astype(np.float32)/16.0
    disparity = stereo.compute(image1,image2,cv2.CV_32F).astype(np.float32)/10.0
    disparity = cv2.GaussianBlur(disparity,(3,3),0)
    return disparity

def main():
    imageM1 = cv2.imread('/Users/vidyach/Desktop/cvip/project2/data/tsucuba_left.png',0)
    colorImage1 =  cv2.imread('/Users/vidyach/Desktop/cvip/project2/data/tsucuba_left.png')
    imageM2 = cv2.imread('/Users/vidyach/Desktop/cvip/project2/data/tsucuba_right.png',0)
    colorImage2 = cv2.imread('/Users/vidyach/Desktop/cvip/project2/data/tsucuba_right.png')
    
    #calling key points functions for Mountain1
    keyPointImageM1,keypointsM1,Descriptor1 = SIFTKeyP(imageM1,imageM1)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_sift1.png",keyPointImageM1)
    
    #calling key points functions for Mountain2
    keyPointImageM2,keypointsM2,Descriptor2 = SIFTKeyP(imageM2,imageM2)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_sift2.png",keyPointImageM2)
    
    #matching the two images
    matchedImage,goodMatches = matchingKeypoints(keypointsM1,keypointsM2,Descriptor1,Descriptor2,imageM1,imageM2)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_matches_knn.png",matchedImage)
    
    #generating fundamental matrix
    FMatrix,image1,image2 = fundamentalMatrix(keypointsM1,keypointsM2,goodMatches,imageM1,imageM2)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_epi_left.png",image1)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_epi_right.png",image2)
    print("fundamental matrix")
    print(FMatrix)
    #computing the disparity 
    disparityImage = disparity(imageM1,imageM2)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project2/temp/task2_disparity.png",disparityImage)
    
    
main()

