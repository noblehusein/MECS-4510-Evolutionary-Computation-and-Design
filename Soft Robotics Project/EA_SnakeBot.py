import concurrent.futures
from math import *
from vpython import *
from decimal import *
import random as rd
import numpy as np
import pandas as pd
import ast
from numpy import genfromtxt

m=0.1
l1=0.1 #meters

k=10000
Configlist=[[0,0,0],[1000,0,0],[20000,0,0],[10000,0.000090,0],[10000,0.000090,np.pi]]
OGfaces=[[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
def Patient0():
    springs=[]
    masses=[]
    n=0.7
    masses.append((m,[0,0,0+n],[0,0,0],[0,0,0]))
    masses.append((m,[0,0,0.1+n],[0,0,0],[0,0,0]))
    masses.append((m,[0.1,0,0+n],[0,0,0],[0,0,0]))
    masses.append((m,[0,0.1,0+n],[0,0,0],[0,0,0]))
    
    springs.append([k,l1,0,1])
    springs.append([k,l1,0,2])
    springs.append([k,l1,0,3])
    
    springs.append([k,sqrt(0.02),1,2])
    springs.append([k,sqrt(0.02),1,3])
    springs.append([k,sqrt(0.02),2,3])
   
    return masses,springs

def Transfusion(masses,springs,faces):
    
    random_face= rd.choice(faces[-3:])
    
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
    
    index=len(masses)-1
    springs.append([k,sqrt(0.02),random_face[0],index])
    springs.append([k,sqrt(0.02),random_face[1],index])
    springs.append([k,sqrt(0.02),random_face[2],index])
    
    faces.append([random_face[0],random_face[1],index])
    faces.append([random_face[1],random_face[2],index])
    faces.append([random_face[0],random_face[2],index])
    faces.remove(random_face)
    
    return masses, springs, faces




def InitialiseBot(OGfaces):
    faces=OGfaces.copy()
    # miawmass,miawspring=Patient0()
    # t=0
    # while t<30:
    #     miawmass,miawspring,faces= Transfusion(miawmass,miawspring,faces)
    #     t=t+1
    # mass=[]
    # spring=[]
    # mass=miawmass.copy()
    # spring=miawspring.copy()
    FinalM=[]
    FinalS=[]
    import csv
    with open('mass2.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    for i in range(0,len(data)):
        j=ast.literal_eval(data[i][0])
        j=list(j)
        FinalM.append(j)
        
    with open('spring2.csv', newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)
    for k in range(0,len(data2)):
        l=ast.literal_eval(data2[k][0])
        l=list(l)
        FinalS.append(l)
        
    
    return FinalM,FinalS,faces


def initializeGraphics():
    
    import vpython
    
    import random
    import numpy as np
    import pandas as pd
    floor = box(length = 10, height = 10, width = 0.0001, color=color.white)
    x_axis = arrow(pos=vector(0,0,0), axis=vector(0.5,0,0),color=color.red)
    y_axis = arrow(pos=vector(0,0,0), axis=vector(0,0.5,0),color=color.green)
    z_axis = arrow(pos=vector(0,0,0), axis=vector(0,0,0.5),color=color.blue)
    masses,springs,__= InitialiseBot(OGfaces)
    massesGL=[0]*len(masses)
    masses_shadowGL=[0]*len(masses)
    for k in range(len(masses)):
        massesGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],masses[k][1][2]), color=color.black)
        # masses_shadowGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],0), color=color.gray(0.5))
       
       
       
    
    springsGL=[0]*len(springs)
    springs_shadowGL=[0]*len(springs)
    for k in range(len(springs)):
        i = springs[k][2]
        j = springs[k][3]
        springsGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
                    axis=vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
                    radius=0.002, color=color.orange)
        springs_shadowGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],0),
                    axis=vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0),
                    radius=0.003, color=color.black, opacity=0.25)
    return massesGL,springsGL,springs_shadowGL
# floor = box(length = 12, height =12, width = 0.0001, color=color.white,texture=textures.wood)
# x_axis = arrow(pos=vector(0,0,0), axis=vector(0.5,0,0),color=color.red)
# y_axis = arrow(pos=vector(0,0,0), axis=vector(0,0.5,0),color=color.green)
# z_axis = arrow(pos=vector(0,0,0), axis=vector(0,0,0.5),color=color.blue)


# miawmass,miawspring,faces=InitialiseBot(OGfaces)
# popsize=1

# master_masses,master_springs=[],[]
# master_masses.append(miawmass)
# master_springs.append(miawspring)

# master_massesGL,master_springsGL,master_shadow=[],[],[]

# for i in range(popsize):
#     masses=master_masses[i]
#     #print(masses)
#     springs=master_springs[i]
#     massesGL=[0]*len(masses)
#     masses_shadowGL=[0]*len(masses)
    
#     for k in range(len(masses)):
#         massesGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],masses[k][1][2]), color=color.magenta)
#         massesGL[k].visible = True
#         # masses_shadowGL[k]=sphere(radius = 0.01, pos = vector(masses[k][1][0],masses[k][1][1],0), color=color.gray(0.5))
#     master_massesGL.append(massesGL)   
       
       
#     springsGL=[0]*len(springs)
#     springs_shadowGL=[0]*len(springs)
#     for k in range(len(springs)):
#         i = springs[k][2]
#         j = springs[k][3]
#         springsGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
#                     axis=vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]),
#                     radius=0.002, color=color.cyan)
#         springs_shadowGL[k]=cylinder(pos=vector(masses[i][1][0],masses[i][1][1],0),
#                     axis=vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0),
#                     radius=0.003, color=color.black, opacity=0.25)
#     master_springsGL.append(springsGL) 
#     master_shadow.append(springs_shadowGL)
    


def CMCalculator(masses):
    L=[]
    m=[]
    for i in range(0,len(masses)):
        L.append(masses[i][1])
        m.append(masses[i][0])
    L=np.array(L);m=np.array(m);
    cm= [sum(L[:,0]*m)/sum(m),sum(L[:,1]*m)/sum(m),sum(L[:,2]*m)/sum(m)]
    return cm
    
    
############################ SIMULATOR ###########################
#global variables
global g
global dt
global T 
global kc
global mu
global mu_k

def Simulation(DNA):

    g = vector(0,0,-9.81) #m/s²
    dt = 0.001 #s (we can increase this or decrease as we want)
    T = 0 #s
    kc = 100000 #stiffness of the ground
    mu=1
    mu_k=0.8
    iter=0
    masses,springs,__=InitialiseBot(OGfaces)
    initialCM=CMCalculator(masses)
    #simulation loop
    t=0    
    while T < 5: 
            poten=0
            elen=0
            kien=0     
            forces = [vector(0,0,0)]*len(masses)
            #print(T)
            for i in range(len(springs)):
                
                springs[i][1]=springs[i][1]+Configlist[(DNA[i])][1]*sin(10*T+Configlist[(DNA[i])][2])
        
            for j in range(len(springs)):
                    i = springs[j][2] #indice masse 
                    k = springs[j][3] #indice de l'autre masse connectée
                    
                    L=sqrt((masses[k][1][0]-masses[i][1][0])**2 +(masses[k][1][1]-masses[i][1][1])**2 + (masses[k][1][2]-masses[i][1][2])**2)
                    
                    l0=springs[j][1]
                    stiff=Configlist[(DNA[j])][0]#springs[j][0]
                    delta_l= L -l0
                    #if abs(delta_l)>0.001:
                    
                    direction= (vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))/mag(vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))
                    
                    forces[i] = forces[i] + stiff*(L-l0)*direction
                    forces[k] = forces[k] - stiff*(L-l0)*direction
                    #else:
                        #forces[i] = forces[i]
                        #forces[k] = forces[k]
                        
            
            
            for i in range(len(masses)):
                forces[i]+= masses[i][0]*g
        
                if masses[i][1][2] < 0:
                    forces[i] = forces[i] - kc*masses[i][1][2]*vector(0,0,1)
                    
                    Fp=vector(forces[i].x,forces[i].y,0)
                    Fn=vector(0,0,forces[i].z)
                    
                    if abs(mag(Fp)) < abs(mag(Fn))*mu:
                        forces[i] = forces[i] - Fp
                    else:
                        forces[i] = forces[i] - abs(mag(Fn))*mu_k*(Fp/abs(mag(Fp)))

            
            for i in range(len(masses)):
              
                  #calculate the new acceleration components with F=ma
                  masses[i][3][0] = forces[i].x/masses[i][0]
                  masses[i][3][1] = forces[i].y/masses[i][0]
                  masses[i][3][2] = forces[i].z/masses[i][0]
                  
                  #update the velocity components with v=v+a*dt
                  masses[i][2][0] = (masses[i][2][0] + masses[i][3][0]*dt)*0.995
                  masses[i][2][1] = (masses[i][2][1] + masses[i][3][1]*dt)*0.995
                  masses[i][2][2] = (masses[i][2][2] + masses[i][3][2]*dt)*0.995
        
                  
                
                  #update the position components with p=p+v*dt
                  masses[i][1][0] += masses[i][2][0]*dt
                  masses[i][1][1] += masses[i][2][1]*dt   
                  masses[i][1][2] += masses[i][2][2]*dt
                  
            #       massesGL[i].visible = False
            #       # masses_shadowGL[i].visible = False
            #       massesGL[i].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #       # masses_shadowGL[i].pos = vector(masses[i][1][0],masses[i][1][1],0)
            #       massesGL[i].visible = True
            #       # masses_shadowGL[i].visible = True
                  
                 
            # for i in DNA:
            #     if i == 0:
            #         springsGL[i].opacity = 0.1
            #     else:
            #         springsGL[i].opacity = 1
                    
            # for k in range(len(springs)):
            
            #     i = springs[k][2]
            #     j = springs[k][3]
   
                
            #     springsGL[k].visible = False
            #     springs_shadowGL[k].visible = False

            #     springsGL[k].pos = vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #     springsGL[k].axis = vector(masses[j][1][0],masses[j][1][1],masses[j][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2])
            #     springs_shadowGL[k].pos = vector(masses[i][1][0],masses[i][1][1],0)
            #     springs_shadowGL[k].axis = vector(masses[j][1][0],masses[j][1][1],0)-vector(masses[i][1][0],masses[i][1][1],0)
            #     springsGL[k].visible = True
            #     springs_shadowGL[k].visible = True
            
            # el.append(elen)
            t=t+dt
            
            
            iter=iter+1                        
            T=T+dt    
    FinalCM=CMCalculator(masses)
    return [initialCM,FinalCM]
    
def Visualize(DNA):
    import concurrent.futures

    import vpython 

    import random as rd
    import numpy as np
    import pandas as pd
    import ast
    g = vector(0,0,-9.81) #m/s²
    dt = 0.001 #s (we can increase this or decrease as we want)
    T = 0 #s
    kc = 100000 #stiffness of the ground
    mu=1
    mu_k=0.8
    iter=0
    masses,springs,__=InitialiseBot(OGfaces)
    massesGL,springsGL,springs_shadowGL=initializeGraphics()
    initialCM=CMCalculator(masses)
    #simulation loop
    t=0    
    while T < 15: 
            poten=0
            elen=0
            kien=0     
            forces = [vector(0,0,0)]*len(masses)
            #print(T)
            for i in range(len(springs)):
                
                springs[i][1]=springs[i][1]+Configlist[(DNA[i])][1]*sin(10*T+Configlist[(DNA[i])][2])
        
            for j in range(len(springs)):
                    i = springs[j][2] #indice masse 
                    k = springs[j][3] #indice de l'autre masse connectée
                    
                    L=sqrt((masses[k][1][0]-masses[i][1][0])**2 +(masses[k][1][1]-masses[i][1][1])**2 + (masses[k][1][2]-masses[i][1][2])**2)
                    
                    l0=springs[j][1]
                    stiff=Configlist[(DNA[j])][0]#springs[j][0]
                    delta_l= L -l0
                    #if abs(delta_l)>0.001:
                    
                    direction= (vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))/mag(vector(masses[k][1][0],masses[k][1][1],masses[k][1][2])-vector(masses[i][1][0],masses[i][1][1],masses[i][1][2]))
                    
                    forces[i] = forces[i] + stiff*(L-l0)*direction
                    forces[k] = forces[k] - stiff*(L-l0)*direction
                    #else:
                        #forces[i] = forces[i]
                        #forces[k] = forces[k]
                        
            
            
            for i in range(len(masses)):
                forces[i]+= masses[i][0]*g
        
                if masses[i][1][2] < 0:
                    forces[i] = forces[i] - kc*masses[i][1][2]*vector(0,0,1)
                    
                    Fp=vector(forces[i].x,forces[i].y,0)
                    Fn=vector(0,0,forces[i].z)
                    
                    if abs(mag(Fp)) < abs(mag(Fn))*mu:
                        forces[i] = forces[i] - Fp
                    else:
                        forces[i] = forces[i] - abs(mag(Fn))*mu_k*(Fp/abs(mag(Fp)))

            
            for i in range(len(masses)):
              
                  #calculate the new acceleration components with F=ma
                  masses[i][3][0] = forces[i].x/masses[i][0]
                  masses[i][3][1] = forces[i].y/masses[i][0]
                  masses[i][3][2] = forces[i].z/masses[i][0]
                  
                  #update the velocity components with v=v+a*dt
                  masses[i][2][0] = (masses[i][2][0] + masses[i][3][0]*dt)*0.995
                  masses[i][2][1] = (masses[i][2][1] + masses[i][3][1]*dt)*0.995
                  masses[i][2][2] = (masses[i][2][2] + masses[i][3][2]*dt)*0.995
        
                  
         
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
                  
                 
            for i in DNA:
                if i == 0:
                    springsGL[i].opacity = 0.1
                else:
                    springsGL[i].opacity = 1
                    
            for k in range(len(springs)):
            
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
            
            # el.append(elen)
            t=t+dt
            
            
            iter=iter+1                        
            T=T+dt    
    FinalCM=CMCalculator(masses)
    return [initialCM,FinalCM]

       
def RS(masses,springs,initialCM):
    Speed=0
    iter=0
    NewSpeedList=[]
    while iter<1000:
       
        DNA=rd.choices([0,1,2,3,4],k = len(miawspring),weights=[0.04,0.24,0.24,0.24,0.24])
        #print(DNA)
        finalCM=Simulation(masses,springs,DNA)
        print(finalCM)
        newSpeed=(np.sqrt(np.sum(np.square(np.array(finalCM)-np.array(initialCM)))))/(np.pi*0.1)
       
        if newSpeed>Speed:
            Speed=newSpeed
            bestDNA=DNA
        masses,springs,faces=InitialiseBot(OGfaces)
        NewSpeedList.append(Speed)
        iter=iter+1
        print(iter,Speed)
        print('length of faces',len(faces))
        print('mass positions',masses[1][1])
           
    return bestDNA,Speed,NewSpeedList



# initialCM=CMCalculator(masses)

# bestDNA,Speed,NewSpeedList=RS(masses,springs,initialCM)



# #Simulation(masses, springs, bestDNA)

# import matplotlib.pyplot as plt
# n=np.array(NewSpeedList)
# r=np.linspace(0,1000,1000)
# plt.plot(r,n)
       


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
  a=0
  b=0
  while  a==b & a<b & a==0 &b==0:
      a= int(rd.random()* len(List1))
      b= int(rd.random()* len(List1))
   

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
    if (rd.random()< murate):
      j= int(rd.random()*len(List))
      if List[j] == 0:
         List[j]=rd.choice((1,2,3,4))
      elif List[j] == 1:
         List[j]=np.random.choice(np.array([0,2,3,4]),p=[0.001, 0.333, 0.333,0.333])
      elif List[j] == 2:
         List[j]=np.random.choice(np.array([0,1,3,4]),p=[0.001, 0.333, 0.333,0.333])        
      elif List[j] == 3:
         List[j]=np.random.choice(np.array([0,1,2,4]),p=[0.001, 0.333, 0.333,0.333])
      elif List[j] == 4:
         List[j]=np.random.choice(np.array([0,1,2,3]),p=[0.001, 0.333, 0.333,0.333]) 
  return List


def MutationFulltoo(Pop,mutationrate):
  finalpop=[]
  length=len(Pop)
  
  for i in range(0,length):
    if i <= int(length/2):
        finalpop.append(Pop[i])
    else:
        mutefellow=mutation(Pop[i],mutationrate)
        finalpop.append(mutefellow)
  
  return finalpop

   
def EA(mutrate,n,iter):
    #if __name__ == '__main__': 
        masses,springs,faces=InitialiseBot(OGfaces)
        pop=createPop(n,springs)
        import csv
        i=0
        print(len(pop))
        masterspeedz=[]
        
        while i< iter:
            speedz=[]
            #print('EOOW')
              
            with concurrent.futures.ProcessPoolExecutor() as executor:
                print('EOOW')                
                for finalCM in executor.map(Simulation,pop):
                    
                    newSpeed=(np.sqrt(np.sum(np.square(np.array(finalCM[1])-np.array(finalCM[0])))))/(np.pi*0.1)
                    
                    speedz.append(newSpeed) 
              
            
            masterspeedz.append(speedz)
           
            pophalf= MateSelect(speedz,pop,n)
           
            popnew=newchildren(pophalf,n)
           
            pop= MutationFulltoo(popnew,mutrate)
           
            #masses,springs,faces=InitialiseBot(OGfaces)
            print('ITERATION NUMBER:', i)
            i=i+1
        
   
        return masterspeedz,pop
def RMHC(masses,springs,murate):
    Speed=0
    iter=0
    NewSpeedList=[]
    DNA=rd.choices([0,1,2,3,4],k = len(springs),weights=[0.04,0.24,0.24,0.24,0.24])
    while iter<3:
       
        DNA=mutation(DNA,murate)
        #print(DNA)
        finalCM=Simulation(DNA)
        print(finalCM)
        newSpeed=(np.sqrt(np.sum(np.square(np.array(finalCM[1])-np.array(finalCM[0])))))/(np.pi*0.1)
       
        if newSpeed>Speed:
            Speed=newSpeed
            bestDNA=DNA
        masses,springs,faces=InitialiseBot(OGfaces)
        NewSpeedList.append(Speed)
        iter=iter+1
        print(iter,Speed)
        print('length of faces',len(faces))
        print('mass positions',masses[1][1])
           
    return bestDNA,Speed,NewSpeedList
    
    
def createPop(n,springs):
    pop=[]
    for i in range(n):
        DNA=rd.choices([0,1,2,3,4],k = len(springs),weights=[0.001,0.24975,0.24975,0.24975,0.24975])
        pop.append(DNA)
    return pop

# masses,springs,__=InitialiseBot(OGfaces)
# n=8
# iter=10
# masteerspeeds,pop=EA(0.005,n,iter)    
    
def main():
    murate=0.1
    masses,springs,__ = InitialiseBot(OGfaces)
    #bestDNA,Speed,NewSpeedList=RMHC(masses,springs,murate)
    n=12
    iter=1000
    # EA = genfromtxt('LOP2.csv', delimiter=',')
    # EA=list(EA)
    # EA=[ int(x) for x in EA ]
    pop=createPop(n,springs)
    # masteerspeeds,pop=EA(0.7,n,iter)    
    # lop=pop[0]
    
    # import csv
    
    # with open('LOP2.csv', 'w') as f: 
    #     write = csv.writer(f) 
    #     for word in lop:
    #             write.writerow([word])
    
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Speed Value')
    
    # l=np.array(masteerspeeds)
    # for i in range(0,iter):
    #     for j in range(0,n):
    #         plt.scatter(i,l[i,n-j-1])
    # curve=[masteerspeeds[i][0] for i in range(len(masteerspeeds))]
    
    # plt.figure(1)
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Speed Value')
    
    # plt.plot([i for i in range(iter)],curve)
    
    
    # import csv
    # with open('learningcurbeEA2.csv', 'w') as f: 
    #     write = csv.writer(f) 
    #     for word in curve:
    #             write.writerow([word])
    #             #write.writerow(NewSpeedList) 










    FinalCM=Visualize(pop[0])
    #return bestDNA,NewSpeedList
    
    return masteerspeeds,pop
    
    
if __name__ == '__main__': 
    #main()
    faces=OGfaces.copy()
    global miawmass
    global miawspring
    miawmass,miawspring=Patient0()
    t=0
    # while t<30:
    #     miawmass,miawspring,faces= Transfusion(miawmass,miawspring,faces)
    #     t=t+1

    import csv
    with open('mass2.csv', 'w') as f: 
        write = csv.writer(f) 
        for word in miawmass:
                write.writerow([word])
     
    with open('spring2.csv', 'w') as f: 
        write = csv.writer(f) 
        for word in miawspring:
                write.writerow([word])
    
    masteerspeeds,pop= main() 


    
    
    
    
    