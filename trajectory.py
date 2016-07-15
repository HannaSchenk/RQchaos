# -*- coding: utf-8 -*-
"""
@author: schenk

Plot Trajectory for a host-parasite system using Replicator Dynamics 
assuming Matching Allele interactions (see a,b,e)
in 3 types of host and 3 types of parasites (n=3)
"""
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import numpy as np
#import math
plt.close("all") # close figures

#==============================================================================
# Variables and settings
#==============================================================================
a=-1. #alpha1
b=0. #alpha2
e=0. #alpha3
mh=np.array([[a,b,e],[e,a,b],[b,e,a]]) #payoff matrix for host
c=1. #factor for payoff matrix
k=1 #for initial condition
h0=np.array([0.5,0.01*k,0.5-0.01*k]) #initial host frequency
p0=np.array([0.5,0.25,0.25]) #initial parasite frequency

generations=300. #time
steps=1001. #steps to plot

n=len(h0) #number of types n
mp=-c*mh #payoff matrix for parasite

#==============================================================================
# Solving of ODE
#==============================================================================
def deriv(y,t):
    h=y[:n]
    p=y[n:]
    h=h/sum(h) #normalisation
    p=p/sum(p) #normalisation
    dH=np.dot(mh,p) #host fitness defined by interaction with parasite
    bh=np.dot(dH,h) #average host fitness
    hdot=h*(dH-bh)  #ODE for host frequency change
    bP=np.dot(mp,h) #parasite fitness defined by interaction with host
    dp=np.dot(bP,p) #average parasite fitness
    pdot=p*(bP-dp)  #ODE for parasite frequency change
    return np.concatenate([hdot,pdot])
time = np.linspace(0.0,generations,steps) # start,end,steps
y = odeint(deriv,np.concatenate([h0,p0]),time)
h=y[:,:n] #host frequencies for all time steps
p=y[:,n:] #parasite frequencies for all time steps

#==============================================================================
# Plot relative abundance (frequencies), time dependent
#==============================================================================
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(12,5)) 
h1, =axes.plot(time,h[:,0],color="lightskyblue",label=r'$h_1$')
h2, =axes.plot(time,h[:,1],color="dodgerblue",label=r'$h_2$')
h3, =axes.plot(time,h[:,2],color="navy",label=r'$h_3$')

p1, =axes.plot(time,p[:,0],color="lightskyblue",linestyle='--',label=r'$p_1$')
p2, =axes.plot(time,p[:,1],color="royalblue",linestyle='--',label=r'$p_2$')
p3, =axes.plot(time,p[:,2],color="navy",linestyle='--',label=r'$p_3$')

axes.set_ylim([0,1])
axes.legend()
axes.set_ylabel('relative abundance',fontsize=18)
axes.set_xlabel('time',fontsize=18)
axes.set_aspect(90)
fig.subplots_adjust(top=1,bottom=0,left=0.07,right=0.98)
fig.savefig("trajectory.pdf")
plt.show()
