#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9  2021
Uses a refining search process to speed up search
Finds the correct value of sigma for a given temperature
Flavor-symmetric (three flavor)
Finite temperature, finite chemical potential 

Set up to be called as outside function of temperature, chem potential, and light quark mass
@author: seanbartz
"""
import numpy as np
from scipy.integrate import odeint
from solveTmu import blackness

#import matplotlib.pyplot as plt

#import time


"LOOK AT LINE 147-148"




# "temperature in MeV"
Temp=300
#light quark mass
ml=50




def chiral(y,u,params):
    chi,chip=y
    v3,v4,zh,q=params
    
    Q=q*zh**3
    
    #phi = -(mu1*zh*u)**2 + (mu1**2+mu0**2)*(zh*u)**2*(1 - np.exp(-(mu2*zh*u)**2))
    "derivative of the dilaton, using exp parameterization"
    #phip= 2*u*zh**2*(mu0**2+np.exp(-(mu2*zh*u)**2)*(mu0**2+mu1**2)*((u*zh*mu2)**2-1) )
    """Fang uses mu sub-g = 440MeV, unto which becomes phi = mu-g^2 * z^2
    thus phip = 2*mu-g^2 * z -> 2*mu-g*u*zh"""
    "mu-g = 440"
    phip = -2*(440**2)*u*(zh**2)
    "blackness function and its derivative, Reissner-Nordstrom metric"
    "This version is for finite temp, finite chemical potential"
    f= 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp= -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    "EOM for chiral field"
    derivs=[chip,
            ((3*f-u*fp+u*f*phip)/(u*f))*chip - (3*chi-3*v3*chi**2-4*v4*chi**3)/(u**2*f)]
            #((3+u**4)/(u-u**5) +phip)*chip - (-3*chi+4*v4*chi**3)/(u**2-u**6) ]
            
    return derivs

def sigmasearch(T,mu,ml):
    "solve for horizon and charge"
    zh,q=blackness(T,mu)
    Q=q*zh**3
    """
    limits of spatial variable z/zh. Should be close to 0 and 1, but 
    cannot go all the way to 0 or 1 because functions diverge there
    """
    ui = 0.01
    uf = 0.999
    "Create the spatial variable mesh"
    umesh=100
    u=np.linspace(ui,uf,umesh)
    
    
    
 
    
    "This is a constant that goes into the boundary conditions"
    eta=np.sqrt(3)/(2*np.pi)
    
    "For the scalar potential in the action"
    "see papers by Bartz, Jacobson"
    #v3= -3 #only needed for 2+1 flavor
    v4 = 8
    v3 = -3
        
    #sigmal=260**3
    params=v3,v4,zh,q
    "blackness function and its derivative, Reissner-Nordstrom metric"
    "This version is for finite temp, finite chemical potential"
    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    
    "stepsize for search over sigma"
    "Note: search should be done over cube root of sigma, here called sl"
    deltasig = 100
    #tic = time.perf_counter()
    minsigma = 100
    maxsigma = 500
    "initialize variable for correct sigma value"
    truesigma = []
    "This version uses a refining method to search"
    "Runs an order of magnitude faster"
    while deltasig > 0.1:
    
        "initial values for comparing test function"
        oldtest=0
        #print(deltasig)
        for sl in range (minsigma,maxsigma,deltasig):
            "values for chiral field and derivative at UV boundary"
            sigmal = sl**3
            UVbound = [ml*eta*zh*ui + sigmal/eta*(zh*ui)**3, ml*eta*zh + 3*sigmal/eta*zh**3*ui**2]
            
            "solve for the chiral field"
            chiFields=odeint(chiral,UVbound,u,args=(params,))
            
            "test function defined to find when the chiral field doesn't diverge"
            "When test function is zero at uf, the chiral field doesn't diverge"
            test = ((-u**2*fp)/f)*chiFields[:,1]-1/f*(3*chiFields[:,0]-3*v3*chiFields[:,0]**2-4*v4*chiFields[:,0]**3)
            testIR = test[umesh-1]#value of test function at uf
            
            "when test function crosses zero, it will go from + to -, or vice versa"
            "This is checked by multiplying by value from previous value of sigma"
            if oldtest*testIR<0: #and chiFields[umesh-1,0]>0:
                #print(oldtest*testIR)
                #print(sl)
                #print(chiFields[umesh-1,0])
                truesigma=sl #save this value
                "new range is +/- deltasig"
                "Need + and - in case true value is right on a multiple of deltasig"
                maxsigma=sl+deltasig
                minsigma=sl-deltasig
                deltasig=int(deltasig*.1)
                break
            oldtest=testIR
        #print(sl)
        "This protects the program from getting hung up"
        "if it reaches the top of the search range, it refines the search"
        "It may not find anything there, but it will eventually terminate"
        if sl == maxsigma-deltasig:
            deltasig=int(deltasig*.1)
    #toc = time.perf_counter()
    #print(f"Found the value in {toc - tic:0.4f} seconds")
    "solve for the chiral field with the correct physical value of sigma"
    
    return truesigma,zh#,chiFields,u
'call the function with arguments temperature, chemical potential, and quark mass'
#(truesigma,zh)=sigmasearch(300,50,50)
#print(truesigma)

#sigmal=sl**3
#UVbound =[ml*eta*zh*ui + sigmal/eta*(zh*ui)**3, ml*eta*zh + 3*sigmal/eta*zh**3*ui**2]
#chiFields=odeint(chiral,UVbound,u,args=(params,))    
#test=((-u**2*fp)/f)*chiFields[:,1]-1/f*(3*chiFields[:,0]-4*v4*chiFields[:,0]**3)

"plot the chiral field and test function for the correct sigma value"
#fig, ax1=plt.subplots()
#plt.plot(u,chiFields[:,0])
#plt.plot(u,test)
#plt.plot(u,ml*eta*zh*u + sigmal/eta*(zh*u)**3)
#plt.show
