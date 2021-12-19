#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 23:36:26 2021

@author: huseinnoble
"""
import concurrent.futures
from math import *
from vpython import *
from decimal import *
import random as rd
import numpy as np
import pandas as pd

#SSpringtypes 0,1---->Bones; 2,3-------> Muscles ook we will figurre this out

global m
m=0.1

def Base():
    springs=[]
    masses=[]
    n=2
    m=0.1
    masses.append((m,[0,0,0+n],[0,0,0],[0,0,0]))
    masses.append((m,[0,0,0.1+n],[0,0,0],[0,0,0]))
    masses.append((m,[0.1,0,0+n],[0,0,0],[0,0,0]))
    masses.append((m,[0,0.1,0+n],[0,0,0],[0,0,0]))
    
    # springs.append([k,l1,0,1])
    # springs.append([k,l1,0,2])
    # springs.append([k,l1,0,3])
    
    # springs.append([k,sqrt(0.02),1,2])
    # springs.append([k,sqrt(0.02),1,3])
    # springs.append([k,sqrt(0.02),2,3])
   
    return masses,springs



class Tetrahedron: 
    def __init__(self, massnumbers,faces,springType,index):
         self.masses = massnumbers 
         self.faces = faces
         self.springType= springType
         self.index=index
  
         
class faces:
    def __init__(self, faces):
         self.faces = faces

def Transfusion(masses,face):
    
    random_face= face
    
    #Calculates positions
    mass1=masses[random_face[0]][1]
    mass1vect= vector(mass1[0],mass1[1],mass1[2])
    
    mass2=masses[random_face[1]][1]
    mass2vect= vector(mass2[0],mass2[1],mass2[2])
    
    mass3=masses[random_face[2]][1]
    mass3vect= vector(mass3[0],mass3[1],mass3[2])
    
    #calculates the centroid
    
    masscentroid= (mass1vect+mass2vect+mass3vect)/3
    
    #normal vector calculations:
    l1= mass2vect-mass1vect; l2=mass3vect-mass1vect
    n= l1.cross(l2)
    n2=l2.cross(l1)
    ncap=n.norm();n2cap=n2.norm()
    masscentroid1= masscentroid + 0.1*ncap
    masscentroid2= masscentroid + 0.1*n2cap
    a=0;b=0
    for i in masses:
        tempvec=vector(i[1][0],i[1][1],i[1][2])
        a= a+ mag(masscentroid1-tempvec)
        b= b+ mag(masscentroid2-tempvec)
    if a>b:
        FinalCentPos=masscentroid1
    else:
        FinalCentPos=masscentroid2
    
    
    masses.append((m,[FinalCentPos.x,FinalCentPos.y,FinalCentPos.z],[0,0,0],[0,0,0]))
    
    index=len(globalmassIndex)
    globalmassIndex.append(index)
    # springs.append([k,sqrt(0.02),random_face[0],index])
    # springs.append([k,sqrt(0.02),random_face[1],index])
    # springs.append([k,sqrt(0.02),random_face[2],index])
    
    faces=[]
    faces.append([random_face[0],random_face[1],index])
    faces.append([random_face[1],random_face[2],index])
    faces.append([random_face[0],random_face[2],index])
    #faces.remove(random_face)
    
    return masses,(face+[index]),faces


Configlist=[[1000,0,0],[20000,0,0],[10000,0.000050,0],[10000,0.000050,np.pi]]

#Will create one of the soluti0ns 

def CreateTree(globalmassIndex,masses):
    iter=0
    L=[0]*100
    
    Tetrahedron1=Tetrahedron([0,1,2,3], [[0,1,2], [0,1,3], [0,2,3], [1,2,3]], 1,0)
    L[0]=(Tetrahedron1)
    L[1]=[Tetrahedron1.faces[0],Tetrahedron1.index]
    L[2]=[Tetrahedron1.faces[1],Tetrahedron1.index]
    L[3]=[Tetrahedron1.faces[2],Tetrahedron1.index]
    L[4]=[Tetrahedron1.faces[3],Tetrahedron1.index]
    
    while (iter<4):
        a=rd.choice(L[2:])
        
        if type(a) == list:
            index=L.index(a)
            #print(a,index,L)
            face=a[0]
            masses,(massnumber),faces= Transfusion(masses,face)
            
            b=Tetrahedron(massnumber, faces, 1,len(globalmassIndex)-4)
            L[index]=(b)
            
            L[3*(index-1)+2]=([b.faces[0],b.index])
            L[3*(index-1)+3]=([b.faces[1],b.index])
            L[3*(index-1)+4]=([b.faces[2],b.index])

            iter=iter+1
            print(iter)
    return L,masses


       
    
globalmassIndex=[0,1,2,3]

masses,springs= Base()

L,masses=CreateTree(globalmassIndex,masses)
    
    

    