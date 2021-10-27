#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 01:22:50 2021

@author: huseinnoble
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import random as rd
import operator
import pandas as pd
from sympy import * 

x = symbols('x')
  
ops = ['+', '-' ,'*' ,'/' ,'sin' ,'cos'] 
val=[-1,-2,-3,-4,-5,-6,-7,-8,-9,1,2,3,4,5,6,7,8,9,'x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x'] 
# 80% chance of including an x rather than 1-9 values. 
# It helps with avoiding a great number of constant functions that aren't really relevant to the search. 

def CalculateFunc(tree):
    copy=tree.copy()
    for i in range(len(copy)-1,1,-2):

        if str(copy[i])== '0' and str(copy[i-1])=='0':
            continue
            
        if str(copy[i])== 'T':
            k=int((i/2)-1);
            copy[k]=str(copy[k])+ '('+'(3.14/180)*'+str(copy[i-1]) +')'
            continue

        k=int((i/2)-1); 
        copy[k]='('+str(copy[i-1])+ str(copy[k])+ str(copy[i])+')'
        
    return copy[0]



def InialiseRandomTree(n):
    tree = [0] * n
    tree[0]= rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0] #Initialising the tree with an operator

    currentNode=0
    doit=True
    while doit==True:
        #print(currentNode)
        if currentNode ==  len(tree)-1:
            break
        if tree[currentNode] not in ops: #Everytime an element is encountered which is not in the Ops
     
            currentNode=currentNode+1   #list it just skips it.
            continue
        tree = AssignNext(tree[currentNode],currentNode,tree,n) #Calling the Assign tree function here.
        if currentNode ==  len(tree)-1:
            break
        currentNode=currentNode+1
    return tree


def AssignNext(value,Index,tree,n):
 
    S = ops+val

    if int(((n-3)/4)-1)<= Index <=int((n-3)/2): #this function exists to truncate the tree to not exceed "15 cell values"

        if value=='sin' or value == 'cos':

            Kid2='T';Kid1= rd.choice(val); Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
            tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
            return tree
        
        if value=='/': #This function is the special case to avoid dividing by 0
            Kid1= rd.choice(val); Kid2 = rd.choice(val); Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
            tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2
            return tree
        if value == '-':
            Kid1=0;Kid2=0;
            while (Kid1==Kid2):
                Kid1= rd.choice(val); Kid2 = rd.choice(val)
            Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
            tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
            return tree
    
        else:
            Kid1Index= 2*Index+1;Kid2Index= 2*Index+2
            Kid1= rd.choice(val); Kid2 = rd.choice(val)
            tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
            return tree

    if 0<= Index <= int((((n-3)/4)-1)/2)-1:
  
      if value=='sin' or value == 'cos': #This function is special case for sin and cos
          Kid2='T';Kid1= rd.choice(ops); Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree

      if value=='/': #This function is the special case to avoid dividing by 0
          Kid1= rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0]; Kid2 = rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0]; Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree
      if value == '-':
          Kid1=0;Kid2=0;
          while (Kid1==Kid2):
            Kid1= rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0]; Kid2 = rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0]
          Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree
      else: 
          Kid1Index= 2*Index+1;Kid2Index= 2*Index+2;
          Kid1= rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0]; Kid2 = rd.choices(ops, weights=(23, 23, 23, 23,4,4), k=1)[0];
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree

    if int((((n-3)/4)-1)/2)<= Index <= int(((n-3)/4)-2):

      if value=='sin' or value == 'cos': #This function is special case for sin and cos
          Kid2='T';Kid1= rd.choice(S); Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree

      if value=='/': #This function is the special case to avoid dividing by 0
          Kid1= rd.choice(S); Kid2 = rd.choice(S); Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree
      if value == '-':
          Kid1=0;Kid2=0;
          while (Kid1==Kid2):
            Kid1= rd.choice(S); Kid2 = rd.choice(S)
          Kid1Index= 2*Index+1; Kid2Index= 2*Index+2;
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree
      else: 
          Kid1Index= 2*Index+1;Kid2Index= 2*Index+2;
          Kid1= rd.choice(S); Kid2 = rd.choice(S);
          tree[Kid1Index]= Kid1; tree[Kid2Index]= Kid2;
          return tree



L_str = []
L_int = []
file = open(r"data.txt",'r')
lines = [line.strip() for line in file]

for line in lines:
    L_str = line.split(', ')
    L_int.append((float(L_str[0]),float(L_str[1])))

X=np.array([L_int[i][0] for i in range(len(L_int))])
Y=np.array([L_int[i][1] for i in range(len(L_int))])


def CalculateYY_fit_large(func, Y=Y, X=X):
      a=10000000000000000
      #print(func)
      f=sympify(func) 
      if f.has(x):
      
          if f.has(zoo):
              return Y,a
          
          g=lambdify(x,f,'numpy')
          func_values=g(X)
          
          if np.sum(func_values)==float('inf'):
              return func_values,a
          
          else:
              mse = ((func_values - Y)**2).mean(axis=None)
              return func_values,mse
      else:
         return Y,a
     
def CalculateYY_fit_small(func, Y=Y, X=X):
      a=10000000000000000
      f=sympify(func) 
      if  f.has(x):
      
            if f.has(zoo):
                return Y,a
            
            
            g=lambdify(x,f,'numpy')
            func_values=g(X)
            if np.sum(func_values)==float('inf'):
                return func_values,a
            else:
                mae = np.sum(abs(func_values - Y))
                return func_values,mae
      else:
        return Y,a

# I plan on using MSE for the first Half to quickly get close to the real path, 
# then use MAE for the remaining iterations to adjust more precisely

def random_search(n):
    curve=[]
    best_tree = InialiseRandomTree()
    best_func = CalculateFunc(best_tree)
    YY,best_fit= CalculateYY_fit_large(best_func)

    for i in range (n):
      print(i)
      if i < n/2 :
          new_tree=InialiseRandomTree()
          new_func=CalculateFunc(new_tree)
          new_YY,new_fitness=CalculateYY_fit_large(new_func)
          if new_fitness < best_fit and new_fitness!=-1:
              best_tree = new_tree
              best_fit = new_fitness
          curve.append(best_fit)  
          
      else: 
          new_tree=InialiseRandomTree()
          new_func=CalculateFunc(new_tree)
          new_YY,new_fitness=CalculateYY_fit_small(new_func)
          if new_fitness < best_fit and new_fitness!=-1:
              print('meilleure fitness apres la moitié')
              best_tree = new_tree
              best_fit = new_fitness
          curve.append(best_fit)  

    return curve,best_tree

#function performs random mutatinons
def random_mutation(tree,index,mutationrate,ops=ops):
    new_tree=tree.copy()
    
    if (rd.random()< mutationrate):
        if (index % 2) == 0:
                if index <=1000:
                    i=rd.randrange(0,15)
                    if new_tree[i] in val[:9]:
                        new_tree[i]='x'
                    elif new_tree[i] == 'x':
                        new_tree[i]=rd.choice(val[:9])    
                    elif new_tree[i] in ops[:4]:
                        new_tree[i]=rd.choice(ops[:4])
                    elif new_tree[i]=='sin':
                         new_tree[i]='cos'
                    elif new_tree[i]=='cos':
                         new_tree[i]='sin'
                elif 1001 <= index <= 2000:
                    i=rd.randrange(15,62)
                    if new_tree[i] in val[:9]:
                        new_tree[i]='x'
                    elif new_tree[i] == 'x':
                        new_tree[i]=rd.choice(val[:9])    
                    elif new_tree[i] in ops[:4]:
                        new_tree[i]=rd.choice(ops[:4])
                    elif new_tree[i]=='sin':
                         new_tree[i]='cos'
                    elif new_tree[i]=='cos':
                         new_tree[i]='sin'
                elif index > 2000:
                    i=rd.randrange(62,255)
                    if new_tree[i] in val[:9]:
                        new_tree[i]='x'
                    elif new_tree[i] == 'x':
                        new_tree[i]=rd.choice(val[:9])    
                    elif new_tree[i] in ops[:4]:
                        new_tree[i]=rd.choice(ops[:4])
                    elif new_tree[i]=='sin':
                         new_tree[i]='cos'
                    elif new_tree[i]=='cos':
                         new_tree[i]='sin'
        else:
            
            i = rd.choice(range(0,124));
            while new_tree[i] not in ops:
                     i = rd.choice(range(0,126))
            a=GetChildren(i, new_tree)                
            p=InialiseRandomTree(len(a));
            k=0
            for  j in a:
                 new_tree[j]=p[k]
                 k=k+1
           
        
    return new_tree



def MutationFulltoo(Population,mutationrate,index):
  finalpop=[]
  for i in range(0,len(Population)):
    mutefellow=random_mutation(Population[i],index,mutationrate)
    finalpop.append(mutefellow)
  return finalpop



def createPop(PopSize):
    pop=[]
    for i in range(0,PopSize):
        pop.append(InialiseRandomTree(255))
    return pop

def MatePoolSelect(pop):
    n= int(len(pop)/2)
    #Takes in the entire population and wiill return the top 50% initially.
    fitnessStorer=[]
    popnew=[]
    for i in range(0,len(pop)):
        
        new_func=CalculateFunc(pop[i])
        new_YY,new_fitness=CalculateYY_fit_large(new_func)
        fitnessStorer.append((new_fitness,i))
      
    df = pd.DataFrame(fitnessStorer, columns = ['fitness','Index'])
    m=df.sort_values(["fitness"], ascending=True)
    miaw= (m.head(n))
    b=miaw.iloc[:,1:].values.T
    for i in range(0,n):
      popnew.append(pop[b[0,i]])
    print(miaw)
    return popnew


def GetChildren(index,tree):
  counter=0
  S=[]
  S.append(index)
  while counter !=1000:
    Kid1Index= 2*S[counter]+1
    Kid2Index= 2*S[counter]+2
    if Kid2Index>=len(tree)-1:
      return S
    S.append(Kid1Index);S.append(Kid2Index)
    counter=counter+1
    
    

def Crossover(P1,P2):
  Parent1,Parent2=P1.copy(),P2.copy()
  i,j=len(Parent1)-1,len(Parent1)-1
  element='T'
  while element == 'T':
      if Parent1[i] in ops and Parent2[i]!='T' and Parent2[i]!=0:
        element='meoaw'
      else:
          i = rd.choice(range(0,126))
  dependents1= GetChildren(i,Parent1)
  for i in dependents1:
      Parent1[i],Parent2[i]=Parent2[i],Parent1[i]
  return Parent1,Parent2

# def Crossover(P1,P2):
#   Parent1,Parent2=P1.copy(),P2.copy()
#   i,j=len(Parent1)-1,len(Parent1)-1
#   element='T'
#   while element == 'T':
#       if Parent1[i] in ops and Parent2[i]!='T' and Parent2[i]!=0:
#         element='meoaw'
#       else:
#          i = rd.choice(range(0,255))

#   #print(Parent1[i],i)
#   #print(Parent2[i],i)
#   dependents1= GetChildren(i,Parent1)
#   #print(dependents1)
#   Child1,Child2=Parent1.copy(),Parent2.copy()
#   for i in dependents1:
#       Child1[i],Child2[i]=Parent2[i],Parent1[i]
#   #print(len(Parent1))
#   #print(len(Parent2))
#   yy_p1,fit_p1=CalculateYY_fit_large(CalculateFunc(Parent1))
#   yy_p2,fit_p2=CalculateYY_fit_large(CalculateFunc(Parent2))
#   yy_c1,fit_c1=CalculateYY_fit_large(CalculateFunc(Child1))
#   yy_c2,fit_c2=CalculateYY_fit_large(CalculateFunc(Child2))
  
#   dp1c1 = abs(fit_c1-fit_p1)
#   dp2c2 = abs(fit_c2-fit_p2)
#   dp1c2 = abs(fit_c2-fit_p1)
#   dp2c1 = abs(fit_c1-fit_p2) 
        
#   if dp1c1+dp2c2 < dp1c2 + dp2c1:
#         if fit_c1 < fit_p1:
#             Parent1=Child1
#         elif fit_c2 < fit_p2:
#             Parent2=Child2
#   else:
#         if fit_c1 < fit_p2:
#             Parent2=Child1
#         elif fit_c2 < fit_p1:
#             Parent1=Child2
      
#   return Parent1,Parent2



def CrosssoverMain(population):
    FinalPop= population.copy()
    print(len(population))
    C=0
    #rd.shuffle(FinalPop)
    for i in range(0,int(len(population)/2)):
        C1,C2= Crossover(FinalPop[C],FinalPop[C+1])
        C=C+2
        FinalPop.append(C1);FinalPop.append(C2)
    return FinalPop





def EA(initialpop,Index,mutationrate):
    
    g= MatePoolSelect(j)
    #print(len(g))
    h= CrosssoverMain(g)
    print(len(h))
    k= MutationFulltoo(h,mutationrate,Index)
    return k
    


def RMHC(n):
    curve=[]
    best_tree = InialiseRandomTree()
    best_func = CalculateFunc(best_tree)
    YY,best_fit= CalculateYY_fit_large(best_func)


    for i in range (n):
        print(i)
        old_tree=best_tree
        old_func=CalculateFunc(old_tree)
        old_YY,old_fitness=CalculateYY_fit_large(old_func)
        
        new_tree= random_mutation(old_tree,i)
        new_func=CalculateFunc(new_tree)
        new_YY,new_fitness=CalculateYY_fit_large(new_func)
        
        if new_fitness < old_fitness and new_fitness !=-1:
            best_tree=new_tree
            best_fit=new_fitness
        else:
            best_tree= old_tree
        curve.append(best_fit)
        
    return curve,best_tree

j=createPop(128)
FitnessBoi=[]
#j=EA(j,i,0.4)
iters=0
accuracy=[]
mutationrate=0.7
while iters<3000:
    
    g= MatePoolSelect(j)
    
    #This Calculates the Fitness of the plot and how it comes  down
    miaw,fitnass=CalculateYY_fit_large(CalculateFunc(g[0]))
    FitnessBoi.append(fitnass)
    print(FitnessBoi)
    
    #This Calculates Accuracy of Plot
    e=lambdify(x,sympify(CalculateFunc(g[0])),'numpy');
    acc=np.sum(abs(Y-np.array(e(X))))/len(Y)
    accuracy.append(1/(acc))
    
    
    #Making thee Gif
    if (iters % 10) == 0:
        plt.title('The '+str(iters)+'th'+' Iterataion')
        plt.plot(X,Y,c='orange') 
        plt.plot(X,e(X),c='blue')
        plt.savefig('Gif/image'+str(iters))
        plt.clf()


    #print(len(g))
    h= CrosssoverMain(g)
    print(len(h))
    k= MutationFulltoo(h,mutationrate,iters)
    print(iters)
    iters=iters+1

    j=k



# curve1,best1=RMHC(5000)
# curve2,best2=RMHC(5000)
# curve3,best3=RMHC(5000)
# curve4,best4=RMHC(5000)

# curve=[(curve1[i]+curve2[i]+curve3[i]+curve4[i])/4 for i in range(len(curve1))]

# curve_err=[0]*5000
# for i in range(1,len(curve1)-1):
#     curve_err[i]=(max((curve1[i],curve2[i],curve3[i],curve4[i]))-min((curve1[i],curve2[i],curve3[i],curve4[i])))

        

#plt.errorbar([i for i in range(5000)], curve, yerr=curve_err, label='Random Search', ecolor='black', capsize=2, errorevery=500)
#plt.legend(loc='upper right')
#plt.ylabel('Fitness (MSE_first_50%-MAE_last_50%)')
#plt.xlabel('Number of Evaluations')
#plt.show()  


YY1,b1=CalculateYY_fit_small(CalculateFunc(j[0]))
# YY2,b2=CalculateYY_fit_small(CalculateFunc(best2))
# YY3,b3=CalculateYY_fit_small(CalculateFunc(best3))
# YY4,b4=CalculateYY_fit_small(CalculateFunc(best4))

plt.plot(X,Y,c='orange',label="Given function")
plt.plot(X,YY1,c='blue',label="Built function 1")
# plt.plot(X,YY2,c='red',label="Built function 2")
# plt.plot(X,YY3,c='black',label="Built function 3")
# plt.plot(X,YY4,c='purple',label="Built function 4")
# plt.legend(loc='upper left')
# plt.show()
print(sympify(CalculateFunc(j[0])))
import csv 
with open('Y-values1.csv', 'w') as f: 
    write = csv.writer(f) 
    for val in YY1:
        write.writerow([val])
        
with open('Func1.csv', 'w') as f: 
    write = csv.writer(f) 
    for val in j[0]:
        write.writerow([val])
        
with open('Accuracy1.csv', 'w') as f: 
    write = csv.writer(f) 
    for val in accuracy:
        write.writerow([val])
        
with open('Fitness1.csv', 'w') as f: 
    write = csv.writer(f) 
    for val in FitnessBoi:
        write.writerow([val])

        