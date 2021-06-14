#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:55:12 2021
Plot the chiral field and test function for given parameters, to examine graphically
This is for the model with three flavors of quarks, with equal mass.
@author: seanbartz
"""

import numpy as np
from scipy.integrate import odeint
from solveTmu import blackness

import matplotlib.pyplot as plt

# "temperature in MeV"
Temp=300.
#light quark mass
ml=50
#chem potential
mu=50

#value of the cube root of chiral condensate in MeV
#386-387 MeV
sl=483



def chiral(y,u,params):
    chi,chip=y

    v3,v4,zh,q,lam,gam,muc=params

    
    Q=q*zh**3
    
    #phi = -(mu1*zh*u)**2 + (mu1**2+mu0**2)*(zh*u)**2*(1 - np.exp(-(mu2*zh*u)**2))
    "derivative of the dilaton, using exp parameterization"
    #phip= 2*u*zh**2*(mu0**2+np.exp(-(mu2*zh*u)**2)*(mu0**2+mu1**2)*((u*zh*mu2)**2-1) )
    """Fang uses mu sub-g = 440MeV, unto which becomes phi = mu-g^2 * z^2
    thus phip = 2*mu-g^2 * z -> 2*mu-g*u*zh"""
    "mu-g = 440"
    phip = -2*(440**2)*u*(zh**2)
    'Reissner-Nordstrom blackness function and its derivative.'
    f= 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp= -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    "EOM for chiral field"
    derivs=[chip,
          -chip * (fp/f + 3/(zh*u) - phip) + 1/(u*f) * (chi*(-3 - muc**2 * zh**2) + chi**3 *lam + chi**2*(gam/(2 * np.sqrt(2))))]          
    return derivs

"solve for horizon and charge"
zh,q=blackness(Temp,mu)
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
#v3= -3 #only needed for 2+1 flavor or 3 flavor
v4 = 8
v3 = -3

"Gamma"
gam=-22.6
    
"Lambda"
lam=16.8

muc=1200

#sigmal=260**3

params=v3,v4,zh,q,lam,gam,muc 

"blackness function and its derivative, Reissner-Nordstrom metric"
"This version is for finite temp, finite chemical potential"
f = 1 - (1+Q**2)*u**4 + Q**2*u**6
fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5

sigmal = sl**3
UVbound = [ml*eta*zh*ui + sigmal/eta*(zh*ui)**3, ml*eta*zh + 3*sigmal/eta*zh**3*ui**2]

"solve for the chiral field"
chiFields=odeint(chiral,UVbound,u,args=(params,))

"test function defined to find when the chiral field doesn't diverge"
"When test function is zero at uf, the chiral field doesn't diverge"
test = ((-u**2*fp)/f)*chiFields[:,1]-1/f*(3*chiFields[:,0]-3*v3*chiFields[:,0]**2-4*v4*chiFields[:,0]**3)
testIR = test[umesh-1]#value of test function at uf

#chiralPot=v3*chiFields[:,0]**3+v4*chiFields[:,0]**4


plt.plot(u,chiFields[:,0])
plt.xlabel(r'$z/z_h$')
plt.ylabel(r'$\chi(z)$')
#plt.plot(u,test)

#plt.plot(u,chiralPot)
