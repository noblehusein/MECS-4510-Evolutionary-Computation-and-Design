# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:43:46 2021

@author: AID-HAMMOU
"""

import concurrent.futures
from math import *
from vpython import *
from decimal import *
import random
import numpy as np
import pandas as pd

getcontext().prec=2

def InitialiseBot():
   
    springs=[]
    masses=[]
   
    #mass object : (m,pos,v,a) with v and a vectors
    #spring object : (k,l0,mass number i,mass number j) with i<j
    n_masses = 8
    m=0.1 #kg
    #pos vector from reference frame on bottom left corner of cube .[]
   
    #for i in range(n_masses): ## need a way to compute pos vectors automatically
   
    masses.append((m,[0.1,0.1,0],[0,0,0],[0,0,0])) #m0
   
    masses.append((m,[0.2,0.1,0],[0,0,0],[0,0,0])) #m1
   
    masses.append((m,[0.1,0.2,0],[0,0,0],[0,0,0])) #m2
   
    masses.append((m,[0.1,0.1,0.1],[0,0,0],[0,0,0])) #m3
   
    masses.append((m,[0.1,0.2,0.1],[0,0,0],[0,0,0])) #m4
   
    masses.append((m,[0.2,0.2,0],[0,0,0],[0,0,0])) #m5
   
    masses.append((m,[0.2,0.1,0.1],[0,0,0],[0,0,0])) #m6
   
    masses.append((m,[0.2,0.2,0.1],[0,0,0],[0,0,0])) #m7
   
    #New cUBE
    masses.append((m,[0.3,0.1,0],[0,0,0],[0,0,0])) #m8
   
    masses.append((m,[0.3,0.2,0],[0,0,0],[0,0,0])) #m9
   
    masses.append((m,[0.3,0.1,0.1],[0,0,0],[0,0,0])) #m10
   
    masses.append((m,[0.3,0.2,0.1],[0,0,0],[0,0,0])) #m11
   
    #new cube
   
    masses.append((m,[0.4,0.1,0],[0,0,0],[0,0,0])) #m12
   
    masses.append((m,[0.4,0.2,0],[0,0,0],[0,0,0])) #m13
   
    masses.append((m,[0.4,0.1,0.1],[0,0,0],[0,0,0])) #m14
   
    masses.append((m,[0.4,0.2,0.1],[0,0,0],[0,0,0])) #m15
   
    #new cube
   
    masses.append((m,[0.5,0.1,0],[0,0,0],[0,0,0])) #m16
   
    masses.append((m,[0.5,0.2,0],[0,0,0],[0,0,0])) #m17
   
    masses.append((m,[0.5,0.1,0.1],[0,0,0],[0,0,0])) #m18
   
    masses.append((m,[0.5,0.2,0.1],[0,0,0],[0,0,0])) #m19
   
   
   
    #newSideCube
   
    masses.append((m,[0.1,0.3,0],[0,0,0],[0,0,0])) #m20
   
    masses.append((m,[0.2,0.3,0],[0,0,0],[0,0,0])) #m21
   
    masses.append((m,[0.1,0.3,0.1],[0,0,0],[0,0,0])) #m22
   
    masses.append((m,[0.2,0.3,0.1],[0,0,0],[0,0,0])) #m23
   
    #newSideCube
   
    masses.append((m,[0.1,0,0],[0,0,0],[0,0,0])) #m24
   
    masses.append((m,[0.2,0,0],[0,0,0],[0,0,0])) #m25
   
    masses.append((m,[0.1,0,0.1],[0,0,0],[0,0,0])) #m26
   
    masses.append((m,[0.2,0,0.1],[0,0,0],[0,0,0])) #m27
   
   
    #newSideCube which is at the back
   
    masses.append((m,[0.4,0.3,0],[0,0,0],[0,0,0])) #m28
   
    masses.append((m,[0.5,0.3,0],[0,0,0],[0,0,0])) #m29
   
    masses.append((m,[0.4,0.3,0.1],[0,0,0],[0,0,0])) #m30
   
    masses.append((m,[0.5,0.3,0.1],[0,0,0],[0,0,0])) #m31
   
   
    #newSideCube which is at the back
   
    masses.append((m,[0.4,0,0],[0,0,0],[0,0,0])) #m32
   
    masses.append((m,[0.5,0,0],[0,0,0],[0,0,0])) #m33
   
    masses.append((m,[0.4,0,0.1],[0,0,0],[0,0,0])) #m34
   
    masses.append((m,[0.5,0,0.1],[0,0,0],[0,0,0])) #m35
    springs=[]
   
    l0=0.1 #meters
    l0_diag_short=sqrt(0.02) #meters
    l0_diag_long=sqrt(0.03) #meters
    k=10000
    l1=l0;l2=l0_diag_short;l3=l0_diag_long
    springs.append([k,l1,0,1]) #s0
    springs.append([k,l1,0,2]) #s1
    springs.append([k,l1,0,3]) #s2
    springs.append([k,l1,3,4]) #s3
    springs.append([k,l1,4,7]) #s4
    springs.append([k,l1,5,7]) #s5
    springs.append([k,l1,1,5]) #s6
    springs.append([k,l1,6,7]) #s7
    springs.append([k,l1,1,6]) #s8
    springs.append([k,l1,3,6]) #s9
    springs.append([k,l1,2,5]) #s10
    springs.append([k,l1,2,4]) #s11
    #short diag springs (12)
    springs.append([k,l2,0,6]) #s12
    springs.append([k,l2,1,3]) #s13
    springs.append([k,l2,3,7]) #s14
    springs.append([k,l2,4,6]) #s15
    springs.append([k,l2,1,7]) #s16
    springs.append([k,l2,5,6]) #s17
    springs.append([k,l2,0,5]) #s18
    springs.append([k,l2,1,2]) #s19
    springs.append([k,l2,2,3]) #s20
    springs.append([k,l2,0,4]) #s21
    springs.append([k,l2,4,5]) #s22
    springs.append([k,l2,2,7]) #s23
    #long diag springs (4)
    springs.append([k,l3,0,7]) #s24
    springs.append([k,l3,3,5]) #s25
    springs.append([k,l3,1,4]) #s26
    springs.append([k,l3,2,6]) #s27
   
   
    #NewCuuube Sprioonng sostom
    springs.append([k,l1,1,8]) #s0
    springs.append([k,l1,6,10]) #s1
    springs.append([k,l1,7,11]) #s2
    springs.append([k,l1,5,9]) #s3
    springs.append([k,l1,8,9]) #s4
    springs.append([k,l1,9,11]) #s5
    springs.append([k,l1,11,10]) #s6
    springs.append([k,l1,10,8])
   
    springs.append([k,l2,1,10]) #s12
    springs.append([k,l2,6,8]) #s13
    springs.append([k,l2,6,11]) #s14
    springs.append([k,l2,7,10]) #s15
    springs.append([k,l2,9,7]) #s16
    springs.append([k,l2,5,11]) #s17
    springs.append([k,l2,8,5]) #s18
    springs.append([k,l2,1,9]) #s19
    springs.append([k,l2,10,9]) #s20
    springs.append([k,l2,8,11]) #s21
   
    springs.append([k,l3,6,9]) #s24
    springs.append([k,l3,1,11]) #s25
    springs.append([k,l3,7,8]) #s26
    springs.append([k,l3,10,5]) #s27
   
   
   
    #NewCuuube Sprioonng sostom-2
    springs.append([k,l1,8,12]) #s0
    springs.append([k,l1,10,14]) #s1
    springs.append([k,l1,11,15]) #s2
    springs.append([k,l1,9,13]) #s3
    springs.append([k,l1,12,13]) #s4
    springs.append([k,l1,13,15]) #s5
    springs.append([k,l1,14,15]) #s6
    springs.append([k,l1,12,14])
   
    springs.append([k,l2,8,14]) #s12
    springs.append([k,l2,10,12]) #s13
    springs.append([k,l2,10,15]) #s14
    springs.append([k,l2,11,14]) #s15
    springs.append([k,l2,13,11]) #s16
    springs.append([k,l2,9,15]) #s17
    springs.append([k,l2,12,9]) #s18
    springs.append([k,l2,8,13]) #s19
    springs.append([k,l2,14,13]) #s20
    springs.append([k,l2,12,15]) #s21
   
    springs.append([k,l3,10,13]) #s24
    springs.append([k,l3,8,15]) #s25
    springs.append([k,l3,11,12]) #s26
    springs.append([k,l3,14,9]) #s27
   
   
    #NewCuuube Sprioonng sostom-2
    springs.append([k,l1,12,16]) #s0
    springs.append([k,l1,14,18]) #s1
    springs.append([k,l1,15,19]) #s2
    springs.append([k,l1,13,17]) #s3
    springs.append([k,l1,16,17]) #s4
    springs.append([k,l1,17,19]) #s5
    springs.append([k,l1,18,19]) #s6
    springs.append([k,l1,16,18])
   
    springs.append([k,l2,12,18]) #s12
    springs.append([k,l2,14,16]) #s13
    springs.append([k,l2,14,19]) #s14
    springs.append([k,l2,15,18]) #s15
    springs.append([k,l2,17,15]) #s16
    springs.append([k,l2,13,19]) #s17
    springs.append([k,l2,16,13]) #s18
    springs.append([k,l2,12,17]) #s19
    springs.append([k,l2,18,17]) #s20
    springs.append([k,l2,16,19]) #s21
   
    springs.append([k,l3,14,17]) #s24
    springs.append([k,l3,12,19]) #s25
    springs.append([k,l3,15,16]) #s26
    springs.append([k,l3,18,13]) #s27
   
   
    #NewCuuube Sprioonng sostom-2 on the side of the first cube
    springs.append([k,l1,2,20]) #s0
    springs.append([k,l1,5,21]) #s1
    springs.append([k,l1,7,23]) #s2
    springs.append([k,l1,4,22]) #s3
    springs.append([k,l1,20,21]) #s4
    springs.append([k,l1,21,23]) #s5
    springs.append([k,l1,22,23]) #s6
    springs.append([k,l1,20,22])
   
    springs.append([k,l2,2,21]) #s12
    springs.append([k,l2,5,20]) #s13
    springs.append([k,l2,5,23]) #s14
    springs.append([k,l2,7,21]) #s15
    springs.append([k,l2,7,22]) #s16
    springs.append([k,l2,4,23]) #s17
    springs.append([k,l2,2,22]) #s18
    springs.append([k,l2,4,20]) #s19
    springs.append([k,l2,20,23]) #s20
    springs.append([k,l2,21,22]) #s21
   
    springs.append([k,l3,2,23]) #s24
    springs.append([k,l3,5,22]) #s25
    springs.append([k,l3,4,21]) #s26
    springs.append([k,l3,7,20]) #s27
   
    #NewCuuube Sprioonng sostom-2 on the side of the first cube
    springs.append([k,l1,0,24]) #s0
    springs.append([k,l1,1,25]) #s1
    springs.append([k,l1,3,26]) #s2
    springs.append([k,l1,6,27]) #s3
    springs.append([k,l1,26,27]) #s4
    springs.append([k,l1,26,24]) #s5
    springs.append([k,l1,24,25]) #s6
    springs.append([k,l1,25,27])
   
    springs.append([k,l2,0,25]) #s12
    springs.append([k,l2,1,24]) #s13
    springs.append([k,l2,1,27]) #s14
    springs.append([k,l2,6,25]) #s15
    springs.append([k,l2,3,27]) #s16
    springs.append([k,l2,6,26]) #s17
    springs.append([k,l2,3,24]) #s18
    springs.append([k,l2,0,26]) #s19
    springs.append([k,l2,25,26]) #s20
    springs.append([k,l2,24,27]) #s21
   
    springs.append([k,l3,0,27]) #s24
    springs.append([k,l3,3,25]) #s25
    springs.append([k,l3,26,1]) #s26
    springs.append([k,l3,6,24]) #s27
   
    #NewCuuube Sprioonng sostom-2 on the side of the lasst cube
    springs.append([k,l1,13,28]) #s0
    springs.append([k,l1,17,29]) #s1
    springs.append([k,l1,19,31]) #s2
    springs.append([k,l1,15,30]) #s3
    springs.append([k,l1,28,29]) #s4
    springs.append([k,l1,29,31]) #s5
    springs.append([k,l1,31,30]) #s6
    springs.append([k,l1,28,30])
   
    springs.append([k,l2,13,29]) #s12
    springs.append([k,l2,17,28]) #s13
    springs.append([k,l2,17,31]) #s14
    springs.append([k,l2,19,29]) #s15
    springs.append([k,l2,19,30]) #s16
    springs.append([k,l2,15,31]) #s17
    springs.append([k,l2,13,30]) #s18
    springs.append([k,l2,15,28]) #s19
    springs.append([k,l2,28,31]) #s20
    springs.append([k,l2,30,29]) #s21
   
    springs.append([k,l3,30,17]) #s24
    springs.append([k,l3,19,28]) #s25
    springs.append([k,l3,15,29]) #s26
    springs.append([k,l3,31,13]) #s27
   
   
    #NewCuuube Sprioonng sostom-2 on the side of the lasst cube
    springs.append([k,l1,12,32]) #s0
    springs.append([k,l1,16,33]) #s1
    springs.append([k,l1,18,35]) #s2
    springs.append([k,l1,14,34]) #s3
    springs.append([k,l1,32,33]) #s4
    springs.append([k,l1,33,35]) #s5
    springs.append([k,l1,34,35]) #s6
    springs.append([k,l1,34,32])
   
    springs.append([k,l2,12,33]) #s12
    springs.append([k,l2,16,32]) #s13
    springs.append([k,l2,18,33]) #s14
    springs.append([k,l2,16,35]) #s15
    springs.append([k,l2,14,35]) #s16
    springs.append([k,l2,18,34]) #s17
    springs.append([k,l2,14,32]) #s18
    springs.append([k,l2,12,34]) #s19
    springs.append([k,l2,34,33]) #s20
    springs.append([k,l2,32,35]) #s21
   
    springs.append([k,l3,12,35]) #s24
    springs.append([k,l3,14,33]) #s25
    springs.append([k,l3,16,34]) #s26
    springs.append([k,l3,18,32]) #s27
   
    return masses,springs

masses,springs= InitialiseBot()


############################ 3D GRAPHICS CODE ####################
# floor = box(length = 4, height = 4, width = 0.0001, color=color.white,texture=textures.wood)
# x_axis = arrow(pos=vector(0,0,0), axis=vector(0.5,0,0),color=color.red)
# y_axis = arrow(pos=vector(0,0,0), axis=vector(0,0.5,0),color=color.green)
# z_axis = arrow(pos=vector(0,0,0), axis=vector(0,0,0.5),color=color.blue)

# massesGL=[0]*len(masses)
# masses_shadowGL=[0]*len(masses)
# for k in range(len(masses)):
#     massesGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],masses[k][1][2]), color=color.magenta)
#     # masses_shadowGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],0), color=color.gray(0.5))
   
   
   

# springsGL=[0]*len(springs)
# springs_shadowGL=[0]*len(springs)
# for k in range(len(springs)):
#     i = springs[k][2]
#     j = springs[k][3]
#     springsGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
#                 axis=vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
#                 radius=0.002, color=color.cyan)
#     springs_shadowGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],0),
#                 axis=vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0),
#                 radius=0.003, color=color.black, opacity=0.25)

# #simulation loop


# running = True
# def Play(b):
#     global running, remember_dt, dt
#     running = not running
#     if running:
#         b.text = "Pause"
#         dt = remember_dt
#     else:
#         b.text = 'Play'
#         remember_dt = dt
#         dt = 0
#     return

# button(text='Pause', pos=scene.title_anchor, bind=Play)



############################ EEvolutishan ####################



def CMCalculator(masses):
    L=[]
    m=[]
    for i in range(0,len(masses)):
        L.append(masses[i][1])
        m.append(masses[i][0])
    L=np.array(L);m=np.array(m);
    cm= [sum(L[:,0]*m)/sum(m),sum(L[:,1]*m)/sum(m),sum(L[:,2]*m)/sum(m)]
    return cm
   
       
       


#global variables
global g
global dt
global T
global kc
global mu
global mu_k
g = vector(0,0,-9.81) #m/s²
dt = 0.01 #s (we can increase this or decrease as we want)
T = 0 #s
kc = 100000 #stiffness of the ground
mu=1
mu_k=0.8

Configlist=[[1000,0,0],[20000,0,0],[10000,0.000050,0],[10000,0.000050,np.pi]]
#simulation loop

time=[];tot=[];pot=[];el=[];ki=[]
t=0
iter=0
miaw=8
initialCM=CMCalculator(masses)
a=np.zeros((miaw,3))
for i in range((miaw)):
    a[i]=np.array(initialCM)
#print(a)
    

def Simulation(DNA):
    T=0
    t=0
    dt = 0.001
    masses,springs=InitialiseBot()
   
    while T < 3*np.pi*0.1:
   
   
            #print('meow')
            forces = [vector(0,0,0)]*len(masses)
           
            for i in range(len(springs)):
               
                springs[i][1]=springs[i][1]+Configlist[(DNA[i])][1]*sin(10*T+Configlist[(DNA[i])][2])
           
       
            for j in range(len(springs)):
                    i = springs[j][2] #indice masse
                    k = springs[j][3] #indice de l'autre masse connectée
                   
                    L=sqrt((masses[k][1][0]-masses[i][1][0])**2 +(masses[k][1][1]-masses[i][1][1])**2 + (masses[k][1][2]-masses[i][1][2])**2)
                   
                    l0=springs[j][1]
                    stiff=Configlist[(DNA[j])][0]#springs[j][0]
                    delta_l= L -l0
       
                    direction= (vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))/mag(vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))
                   
                    forces[i] = forces[i] + stiff*(L-l0)*direction
                    forces[k] = forces[k] - stiff*(L-l0)*direction
                       
           
           
            for i in range(len(masses)):
                forces[i]+= masses[i][0]*g
       
                if masses[i][1][2] < 0:
                    forces[i] = forces[i] - kc*masses[i][1][2]*vector(0,0,1)
                   
                    Fp=vector(forces[i].x,forces[i].y,0)
                    Fn=vector(0,0,forces[i].z)
                   
                    if mag(Fp) < mag(Fn)*mu:
                        forces[i] = forces[i] - Fp
                    else:
                        forces[i] = forces[i] - mag(Fn)*mu_k*(Fp/mag(Fp))
                   
     
           
            for i in range(len(masses)):
             
                  #calculate the new acceleration components with F=ma
                  masses[i][3][0] = forces[i].x/masses[i][0]
                  masses[i][3][1] = forces[i].y/masses[i][0]    
                  masses[i][3][2] = forces[i].z/masses[i][0]
                 
                  #update the velocity components with v=v+a*dt
                  masses[i][2][0] += masses[i][3][0]*dt*0.8
                  masses[i][2][1] += masses[i][3][1]*dt*0.8  
                  masses[i][2][2] += masses[i][3][2]*dt*0.8
               
                  #update the position components with p=p+v*dt
                  masses[i][1][0] += masses[i][2][0]*dt
                  masses[i][1][1] += masses[i][2][1]*dt  
                  masses[i][1][2] += masses[i][2][2]*dt
                 
                  # massesGL[i].visible = False
                  # # masses_shadowGL[i].visible = False
                  # massesGL[i].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
                  # # masses_shadowGL[i].pos = vector(masses[i][1][0],masses[i][1][1],0)
                  # massesGL[i].visible = True
                  # # masses_shadowGL[i].visible = True
                 
                 
                  #Calculating Potential Energy for all the masses=
                 
                 
                  # poten= poten+  masses[i][0]*(9.81)*(masses[i][1][2])
                  # kien=kien +(0.5)*masses[i][0]*(masses[i][2][2])**2
            # pot.append(poten)
            # ki.append(kien)
           
            # if iter %10==0:
               
            #     for i in range(len(masses)):
            #         massesGL[i].visible = False
            #         # masses_shadowGL[i].visible = False
            #         massesGL[i].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #         # masses_shadowGL[i].pos = vector(masses[i][1][0],masses[i][1][1],0)
            #         massesGL[i].visible = True
            #         # masses_shadowGL[i].visible = True
           
            #     for k in range(len(springs)):
               
            #         i = springs[k][2]
            #         j = springs[k][3]
                   
            #         # L=sqrt((masses[j][1][0]-masses[i][1][0])**2 +(masses[j][1][1]-masses[i][1][1])**2 + (masses[j][1][2]-masses[i][1][2])**2)
                       
            #         # l0=springs[k][1]
            #         # stiff=springs[k][0]
            #         # delta_l= L -l0
            #         # #Calculating Kinetic Energy for all springs:
            #         # elen=elen+ (0.5)*stiff*delta_l**2
                   
                   
            #         springsGL[k].visible = False
            #         springs_shadowGL[k].visible = False
            #         springsGL[k].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #         springsGL[k].axis = vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #         springs_shadowGL[k].pos = vector(masses[i][1][0],masses[i][1][1],0)
            #         springs_shadowGL[k].axis = vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0)
            #         springsGL[k].visible = True
            #         springs_shadowGL[k].visible = True
               
            # el.append(elen)
            t=t+dt
            #print(t)
            # for k in range(28):
            #     #print(type(springs[k][1]))
            #     #if k<=11:
            #         #springs[k][1]=l0+0.01*np.sin(t/2)
            #     if 12<=k<=23:
            #         springs[k][1]=l0_diag_short+0.01*np.sin(100*t)
            #     if 24<=k<=27:
            #         springs[k][1]=l0_diag_long+0.01*np.sin(100*t)
            # print(springs[14][1])
            # tot.append(poten+elen+kien)0
            # #print(tot[-1],poten,elen,ki)
            # time.append(t)
           

                                       
            T=T+dt
    #c
    L=[]
    m=[]
    for i in range(0,len(masses)):
        L.append(masses[i][1])
        m.append(masses[i][0])
    L=np.array(L);m=np.array(m);
    cm= [sum(L[:,0]*m)/sum(m),sum(L[:,1]*m)/sum(m),sum(L[:,2]*m)/sum(m)]
    return cm
    #return finalCM

   



       
def RS(masses,springs):
    Speed=0
    iter=0
    NewSpeedList=[]
    while iter<1000:
       
        DNA=random.choices([0,1,2,3], k = len(springs))
        #print(DNA)
        finalCM=Simulation(masses, springs,DNA)
        newSpeed=(np.sqrt(np.sum(np.square(np.array(finalCM)-np.array(initialCM)))))/(np.pi*0.1)
       
        if newSpeed>Speed:
            Speed=newSpeed
            bestDNA=DNA
        masses,springs=InitialiseBot()
        NewSpeedList.append(Speed)
        iter=iter+1
        print(iter,Speed)
           
    return bestDNA,Speed,NewSpeedList

# bestDNA,Speed,NewSpeedList=RS(masses,springs)



# #Simulation(masses, springs, bestDNA)

# import matplotlib.pyplot as plt
# n=np.array(NewSpeedList)
# r=np.linspace(0,1000,1000)
# plt.plot(r,n)
       

def createPop(n):
    pop=[]
    for i in range(n):
        DNA=random.choices([0,1,2,3], k = len(springs))
        pop.append(DNA)
    return pop

def MateSelect(speedz,DNA,n):
   
    #L=np.array(List)
   
    dist=[]
    distnew=[]
    print(n)
    for i in range(0, n):
        dist.append((speedz[i],i))
    #print(dist[4])
    df = pd.DataFrame(dist, columns = ['speedz','Index'])
    m=df.sort_values(["speedz"], ascending=False)
   
    n= int(n/2)
    miaw=m.head(n)
    b=miaw.iloc[:,1:].values.T
    for i in range(0,n):
      distnew.append(DNA[b[0,i]])
    print(miaw)
   
    return distnew


def breed(List1,List2):
  bachamain1=[]
  bacha1=[]
  bacha2=[]
 
 
  a= int(random.random()* len(List1))
   

  L=np.array(List1);L2=np.array(List2)
 
  bacha1=list(L[0:a])
  bacha2=list(L2[a:])
  #print(L,L2)
 


  bachamain1 =bacha1+bacha2
  print(len(bachamain1))


  return bachamain1

def newchildren(Population,popsize):
  childs=[]
 
  for i in range(0, int(popsize/2)):
    k= breed(Population[i],Population[i+1])
    Population.append(k)
  return Population


def mutation(List,murate):
  for i in range(len(List)):
    if (random.random()< murate):
      j= int(random.random()*len(List))
      if List[j] == 0:
         List[j]=random.choice((1,2,3))
      elif List[j] == 1:
         List[j]=random.choice((0,2,3))
      elif List[j] == 2:
         List[j]=random.choice((0,1,3))         
      elif List[j] == 3:
         List[j]=random.choice((1,2,0)) 
  return List


def MutationFulltoo(Pop,mutationrate):
  finalpop=[]
  for i in range(0,len(Pop)):
    mutefellow=mutation(Pop[i],mutationrate)
    finalpop.append(mutefellow)
  return finalpop

   
def EA(mutrate,n,iter):
    if __name__ == '__main__': 
        masses,springs=InitialiseBot()
        pop=createPop(n)
        i=0
        masterspeedz=[]
        print('goess beforre while loopp')
        while i< iter:
            speedz=[]
            #print('EOOW')
            print('iinside the while loop before cooncuurrence')
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                
                print('EOOW')
                finalCM=[executor.submit(Simulation, popi) for popi in pop]
                #for finalCM in executor.map(Simulation,pop):
                #finalCM = executor.map(Simulation,pop)
                for f in concurrent.futures.as_completed(finalCM):
                    #print(len(pop))
                    #print(f.result(), initialCM)
                    newSpeed=(np.sqrt(np.sum(np.square(np.array(f.result())-np.array(initialCM)))))/(np.pi*0.1)
                    speedz.append(newSpeed) 
            print('went to tthe EA')  
            masterspeedz.append(speedz)
           
            pophalf= MateSelect(speedz,pop,n)
           
            popnew=newchildren(pophalf,n)
           
            pop= MutationFulltoo(popnew,mutrate)
           
            masses,springs=InitialiseBot()
            print('ITERATION NUMBER:', i)
            i=i+1
        
   
        return masterspeedz,pop
       
masses,springs=InitialiseBot()
n=8
iter=10
masteerspeeds,pop=EA(0.005,n,iter)    

# # '''   

# Use thiis to plot the fkn dot ploots
# l=np.array(masteerspeeds)
# for i in range(0,iter):
#     for j in range(0,n):
#         plt.scatter(i,l[i,n-j],s=1)
# '''     
           
EA(0.001,8,10)    
   
   

#Fitness
     