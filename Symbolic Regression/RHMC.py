#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:21:56 2021

@author: huseinnoble
"""
# -- coding: utf-8 --
"""
Created on Wed Oct 20 13:19:51 2021

@author: AID-HAMMOU
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import random as rd
import operator

# import sympy
from sympy import * 

x = symbols('x')
  
ops = ['+', '-' ,'*' ,'/' ,'sin' ,'cos'] 
val=[1,2,3,4,5,6,7,8,9,'x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x'] 
# 80% chance of including an x rather than 1-9 values. 
# It helps with avoiding a great number of constant functions that aren't really relevant to the search. 

def CalculateFunc(tree):
    copy=tree.copy()
    print(copy)
    for i in range(len(copy)-1,1,-2):
        #print(i)
        #if i==0:
            #iter = 14
        #else:
            #iter = len(tree)-2-i
        #print(iter)
        if str(copy[i])== '0' and str(copy[i-1])=='0':
            continue
            
        if str(copy[i])== 'T':
            k=int((i/2)-1);
            #print(k)
            copy[k]=str(copy[k])+ '('+'(3.14/180)*'+str(copy[i-1]) +')'
            #print(tree)
            continue

        k=int((i/2)-1);
        #print(k)
   
        copy[k]='('+str(copy[i-1])+ str(copy[k])+ str(copy[i])+')'
        
            
    return copy[0]

def InialiseRandomTree():
    tree = [0] * 255
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
        tree = AssignNext(tree[currentNode],currentNode,tree) #Calling the Assign tree function here.
        if currentNode ==  len(tree)-1:
            break
        currentNode=currentNode+1
    return tree

def AssignNext(value,Index,tree):
 
    S = ops+val

    if 62<= Index <=126: #this function exists to truncate the tree to not exceed "15 cell values"

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

    if 0<= Index <= 30:
  
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

    if 31<= Index <= 61:

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
      a=-1
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
      a=-1
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

def random_mutation(tree,index,ops=ops):
    new_tree=tree.copy()
    
    
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
    return new_tree


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


curve1,best1=RMHC(5000)
curve2,best2=RMHC(5000)
curve3,best3=RMHC(5000)
curve4,best4=RMHC(5000)

curve=[(curve1[i]+curve2[i]+curve3[i]+curve4[i])/4 for i in range(len(curve1))]

curve_err=[0]*5000
for i in range(1,len(curve1)-1):
    curve_err[i]=(max((curve1[i],curve2[i],curve3[i],curve4[i]))-min((curve1[i],curve2[i],curve3[i],curve4[i])))

        

plt.errorbar([i for i in range(5000)], curve, yerr=curve_err, label='Random Search', ecolor='black', capsize=2, errorevery=500)
plt.legend(loc='upper right')
plt.ylabel('Fitness (MSE_first_50%-MAE_last_50%)')
plt.xlabel('Number of Evaluations')
plt.show()  


YY1,b1=CalculateYY_fit_small(CalculateFunc(best1))
YY2,b2=CalculateYY_fit_small(CalculateFunc(best2))
YY3,b3=CalculateYY_fit_small(CalculateFunc(best3))
YY4,b4=CalculateYY_fit_small(CalculateFunc(best4))

plt.plot(X,Y,c='orange',label="Given function")
plt.plot(X,YY1,c='blue',label="Built function 1")
plt.plot(X,YY2,c='red',label="Built function 2")
plt.plot(X,YY3,c='black',label="Built function 3")
plt.plot(X,YY4,c='purple',label="Built function 4")
plt.legend(loc='upper left')
plt.show()