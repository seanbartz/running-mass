#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9 16:00:51 2021
Gets sigma values for a variety of temperatures
At a particular chemical potential
And a particular quark mass
@author: seanbartz
"""
import numpy as np
from threeflavorRefinedmu import sigmasearch
import matplotlib.pyplot as plt

"light quark mass"
quarkmass=30
"quark chemical potential"
chemPotential=30

numtemps=20
temps=np.linspace(182,188,numtemps)

sigmaArray=np.zeros(numtemps)

for i in range(0,numtemps):
    #print(temps[i])
    (sigmaArray[i],zh)=sigmasearch(temps[i],chemPotential,quarkmass)
    # if i % 10 ==0:
    #     print(temps[i])
    "Stop the scan if we are basically at zero"
    if sigmaArray[i]<=20:
        break

plt.scatter(temps,sigmaArray)
plt.xlabel('Temperature (MeV)')
plt.ylabel(r'$\sigma^{1/3}$ (MeV)')
plt.title(r'$m_q=%i$ MeV, $\mu=%i$ MeV' %(quarkmass,chemPotential))
#plt.scatter(temps,np.gradient(sigmaArray))

print('The largest change in sigma occurs at T =',temps[np.argmin(np.gradient(sigmaArray))])
