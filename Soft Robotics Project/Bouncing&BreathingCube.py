# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 00:38:50 2021

@author: AID-HAMMOU
"""
from math import *
from vpython import *
import numpy as np

springs=[]
springs_shadow=[]
masses=[]
masses_shadow=[]

#mass object : (m,pos,v,a) with v and a vectors 
#spring object : (k,l0,mass number i,mass number j) with i<j
n_masses = 8
m=0.1 #kg
#pos vector from reference frame on bottom left corner of cube .[]

#for i in range(n_masses): ## need a way to compute pos vectors automatically

############################ MASSES ########################################

masses.append((m,[0.1,0.1,0.1],[0,0,0],[0,0,0])) #m0

masses.append((m,[0.2,0.1,0.1],[0,0,0],[0,0,0])) #m1

masses.append((m,[0.1,0.2,0.1],[0,0,0],[0,0,0])) #m2

masses.append((m,[0.1,0.1,0.2],[0,0,0],[0,0,0])) #m3

masses.append((m,[0.1,0.2,0.2],[0,0,0],[0,0,0])) #m4

masses.append((m,[0.2,0.2,0.1],[0,0,0],[0,0,0])) #m5

masses.append((m,[0.2,0.1,0.2],[0,0,0],[0,0,0])) #m6

masses.append((m,[0.2,0.2,0.2],[0,0,0],[0,0,0])) #m7


############################ SPRINGS #######################################

l0=0.1 #meters
l0_diag_short=sqrt(0.02) #meters
l0_diag_long=sqrt(0.03) #meters

l1=l0
l2= l0_diag_short
l3= l0_diag_long
k=1000
#short springs (12)
springs=[]
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
springs.append([k,l3,2,6]) #
############################ 3D GRAPHICS CODE ####################
mybox = box(length = 15, height = 15, width = 0.0001, color=color.white, texture=textures.wood)
x_axis = arrow(pos=vector(0,0,0), axis=vector(0.5,0,0),color=color.red)
y_axis = arrow(pos=vector(0,0,0), axis=vector(0,0.5,0),color=color.green)
z_axis = arrow(pos=vector(0,0,0), axis=vector(0,0,0.5),color=color.blue)

massesGL=[0]*8
masses_shadowGL=[0]*8
for k in range(8):
    massesGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],masses[k][1][2]), color=color.magenta)
    #masses_shadowGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],0), color=color.gray(0.5))
    
springsGL=[0]*28
springs_shadowGL=[0]*28
for k in range(28):
    i = springs[k][2]
    j = springs[k][3]
    springsGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
                axis=vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]), 
                radius=0.002, color=color.cyan)
    springs_shadowGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],0),
                axis=vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0),
                radius=0.003, color=color.black, opacity=0.25)
        



 
############################ SIMULATOR ###########################
#global variables
global g
global dt
global T 
global kc
g = vector(0,0,-9.81) #m/s²
dt = 0.0001 #s (we can increase this or decrease as we want)
T = 0 #s
kc = 100000 #stiffness of the ground
#simulation loop


running = True
def Play(b):
    global running, remember_dt, dt
    running = not running
    if running:
        b.text = "Pause"
        dt = remember_dt
    else:
        b.text = 'Play'
        remember_dt = dt
        dt = 0
    return

button(text='Pause', pos=scene.title_anchor, bind=Play)
t=0
while T < 1: 
    
        forces = [vector(0,0,0)]*8
        
        for j in range(28):
                i = springs[j][2] #indice masse 
                k = springs[j][3] #indice de l'autre masse connectée
                
                L=sqrt((masses[k][1][0]-masses[i][1][0])**2 +(masses[k][1][1]-masses[i][1][1])**2 + (masses[k][1][2]-masses[i][1][2])**2)
                
                l0=springs[j][1]
                stiff=springs[j][0]
                delta_l= L -l0
    
                direction= (vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))/mag(vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))
                
                forces[i] = forces[i] + stiff*(L-l0)*direction
                forces[k] = forces[k] - stiff*(L-l0)*direction
                    
        for i in range(8):
            if t>0.5:
                forces[i]+= masses[i][0]*g

            if masses[i][1][2] < 0:
                forces[i] = forces[i] - kc*masses[i][1][2]*vector(0,0,1)
                
                
        for i in range(8):
          
              #calculate the new acceleration components with F=ma
              masses[i][3][0] = forces[i].x/masses[i][0]
              masses[i][3][1] = forces[i].y/masses[i][0]    
              masses[i][3][2] = forces[i].z/masses[i][0] 
              
              #update the velocity components with v=v+a*dt
              masses[i][2][0] += masses[i][3][0]*dt
              masses[i][2][1] += masses[i][3][1]*dt   
              masses[i][2][2] += masses[i][3][2]*dt
            
              #update the position components with p=p+v*dt
              masses[i][1][0] += masses[i][2][0]*dt
              masses[i][1][1] += masses[i][2][1]*dt   
              masses[i][1][2] += masses[i][2][2]*dt
              
              massesGL[i].visible = False
              # masses_shadowGL[i].visible = False
              massesGL[i].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
              # masses_shadowGL[i].pos = vector(masses[i][1][0],masses[i][1][1],0)
              massesGL[i].visible = True
              # masses_shadowGL[i].visible = True

        
        for k in range(28):
            i = springs[k][2]
            j = springs[k][3]
            
            springsGL[k].visible = False
            springs_shadowGL[k].visible = False
            springsGL[k].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            springsGL[k].axis = vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            springs_shadowGL[k].pos = vector(masses[i][1][0],masses[i][1][1],0)
            springs_shadowGL[k].axis = vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0)
            springsGL[k].visible = True
            springs_shadowGL[k].visible = True
        t=t+dt
        
        for k in range(28):
                #print(type(springs[k][1]))
                #if k<=11:
                    #springs[k][1]=l0+0.0001*np.sin(((2*np.pi/(1/dt))*(t)))
        #print(springs[1][1])
                if 12<=k<=23:
                    springs[k][1]=l0_diag_short+0.05*np.sin(100*t)
                if 24<=k<=27:
                    springs[k][1]=l0_diag_long+0.05*np.sin(100*t)
        print(springs[12][1])
                                    
        T=T+dt    


