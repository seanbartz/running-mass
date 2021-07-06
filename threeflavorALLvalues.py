#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 16  2021
For a given quark mass and chemical potential, 
solves for all sigma values for a range of temperatures.
If there are multiple values, then the transition is 1st order.
@author: seanbartz
"""
import numpy as np
from scipy.integrate import odeint
from solveTmu import blackness

import matplotlib.pyplot as plt

#import time


#light quark mass
quarkmass=22

#chemical potential
chempotential=0

tmin=148
tmax=150
numtemp=40


def chiral(y,u,params):
    global chi,chip
    chi,chip=y

    zh,q,lam,gam,mu_c=params

    Q=q*zh**3
    
    #phi = -(mu1*zh*u)**2 + (mu1**2+mu0**2)*(zh*u)**2*(1 - np.exp(-(mu2*zh*u)**2))
    "derivative of the dilaton, using exp parameterization"
    #phip= 2*u*zh**2*(mu0**2+np.exp(-(mu2*zh*u)**2)*(mu0**2+mu1**2)*((u*zh*mu2)**2-1) )
    """Fang uses mu sub-g = 440MeV, unto which becomes phi = mu-g^2 * z^2
    thus phip = 2*mu-g^2 * z -> 2*mu-g*u*zh"""
    mu_g = 440 
    phip = 2*(mu_g**2)*u*(zh**2)
    f= 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp= -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    "EOM for chiral field"
    derivs=[chip,
          -(chip * (fp/f - 3/u - phip) - 1/(u**2*f) * (chi*(-3 - mu_c**2 * (u*zh)**2) + chi**3 *lam + chi**2*(gam/(2 * np.sqrt(2)))))]          
    return derivs

def sigmasearch(T,mu,ml):
    "strange quark mass same as light quark mass for 3 flavor"
    ms=ml
    
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
    #zeta, normalization constant which is defined by n_c, number of colors
    n_c = 3
    zeta = np.sqrt(n_c)/(2*np.pi)
    
    "Gamma is the coefficient of the cubic term in the scalar potential in the Fang paper"
    gam = -22.6
    "Lambda is the coefficient of the quartic term in the scalar potential in the Fang paper"
    lam=16.8
    
    "mu_c sets the scale of the running scalar mass term in the Fang paper"
    mu_c=1200 
    
    "Dilaton parameter needed in boundary conditions"
    mu_g=440
        
    params=zh,q,lam,gam,mu_c

    "blackness function and its derivative, Reissner-Nordstrom metric"
    "This version is for finite temp, finite chemical potential"
    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    
    "stepsize for search over sigma"
    "Note: search should be done over cube root of sigma, here called sl"
    deltasig = 1
    #tic = time.perf_counter()
    minsigma = 0
    maxsigma = 500
    truesigma = 0
    "This version steps over all values to find multiple solutions at some temps"
    
    "initial values for comparing test function"
    oldtest=0
    j=0
    truesigma=np.zeros(3)

    for sl in range (minsigma,maxsigma,deltasig):
    
        "values for chiral field and derivative at UV boundary"
        sigmal = sl**3
        #z is very VERY close to zero
        #(old) UVbound = [ml*eta*zh*ui + sigmal/eta*(zh*ui)**3, ml*eta*zh + 3*sigmal/eta*zh**3*ui**2]
        #new UVbound from Fang eqn. 11, [chi, chip]
        UVbound = [ml*zeta*(zh*ui)-((ml*ms*gam*zeta**2)/(2*np.sqrt(2)))*(zh*ui)**2 + (sigmal/zeta)*(zh*ui)**3 + 0.0625*ml*zeta*((-ms**2)*(gam**2)*(zeta**2) - (ml**2)*(gam**2)*(zeta**2) + 8*(ml**2)*(zeta**2)*gam + 16*(mu_g**2) - 8*(mu_c**2)) * ((zh*ui)**3)*np.log(zh*ui),   
                   ml*zeta*(zh)-2*((ml*ms*gam*zeta**2)/(2*np.sqrt(2)))*(zh**2*ui) + 3*(sigmal/zeta)*(zh**3*ui**2) + 0.0625*ml*zeta*((-ms**2)*(gam**2)*(zeta**2) - (ml**2)*(gam**2)*(zeta**2) + 8*(ml**2)*(zeta**2)*gam + 16*(mu_g**2) - 8*(mu_c**2)) * zh**3*(3*ui**2*np.log(zh*ui)+ui**2)]

        "solve for the chiral field"
        chiFields=odeint(chiral,UVbound,u,args=(params,))
        
        "test function defined to find when the chiral field doesn't diverge"
        "When test function is zero at uf, the chiral field doesn't diverge"
        test = ((-u**2*fp)/f)*chiFields[:,1]-1/f*(3*chiFields[:,0]+lam*chiFields[:,0]**2+gam/(2*np.sqrt(2))*chiFields[:,0]**3)
        testIR = test[umesh-1]#value of test function at uf
        
        "when test function crosses zero, it will go from + to -, or vice versa"
        "This is checked by multiplying by value from previous value of sigma"
        if oldtest*testIR<0: #and chiFields[umesh-1,0]>0:
            if j<3:
                truesigma[j]=sl #save this value
                j=j+1 #if there are other sigma values, they will be stored also
                #print(truesigma)
            
        oldtest=testIR

    
    return truesigma
temps=np.linspace(tmin,tmax,numtemp)
#need up to 3 sigma values per temperature
truesigma=np.zeros([numtemp,3])

for i in range (0,numtemp):
    truesigma[i,:]=sigmasearch(temps[i],chempotential,quarkmass)
    
plt.scatter(temps,truesigma[:,0])
plt.scatter(temps,truesigma[:,1])
plt.scatter(temps,truesigma[:,2])
minsigma=min(truesigma[:,0])-5
maxsigma=max(truesigma[:,0])+5
plt.ylim([minsigma,maxsigma])
plt.xlabel('Temperature (MeV)')
plt.ylabel(r'$\sigma^{1/3}$ (MeV)')
plt.title(r'$m_q=%i$ MeV, $\mu=%i$ MeV' %(quarkmass,chempotential))

if max(truesigma[:,1])==0:
    print("Crossover or 2nd order")
else:
    print("First order")    
    
    
