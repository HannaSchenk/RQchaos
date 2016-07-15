# -*- coding: utf-8 -*-
"""
@author: schenk

Plot 3-simplex for replicator dynamics with two species (host, parasite).
Three types each. Matching allele model (a=-1, b=e=0)
"""
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import numpy as np
import math

plt.close("all") # close figures

#==============================================================================
# Variables and settings
#==============================================================================
a=-1.
b=0.
e=0.
mh=np.array([[a,b,e],[e,a,b],[b,e,a]]) #payoff matrix for host
c=1. #factor
k=1. #initial condition (1 or 10)
cm = plt.cm.get_cmap('viridis')
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,6)) #figure for host simplex

for k in [1,20]: #for two initial conditions different number of generations
    if k==1:
       generations=10000
       steps=1000000.
    else:
       generations=5000
       steps=500000.
    time = np.linspace(0.0,generations,steps) # start,end,steps 
    h0=np.array([0.5,0.01*k,0.5-0.01*k]) #initial host frequencies
    p0=np.array([0.5,0.25,0.25]) #initial parasite frequencies

    n=len(h0) #number of hosts or parasites
    mp=-c*mh #payoff matrix for parasites

#==============================================================================
# Solving of ODE
#==============================================================================

    def deriv(y,t):
        h=y[:n] #host frequencies
        p=y[n:] #parasite frequencies
        h=h/sum(h) #normalisation
        p=p/sum(p) #normalisation
        dH=np.dot(mh,p) #host fitness influenced by parasites
        bh=np.dot(dH,h) #average host fitness
        hdot=h*(dH-bh)  #ODE for hosts
        bP=np.dot(mp,h) #parasite fitness influenced by host
        dp=np.dot(bP,p) #aberage parasite fitness
        pdot=p*(bP-dp)  #ODE for parasites
        return np.concatenate([hdot,pdot]) #all ODEs

    y = odeint(deriv,np.concatenate([h0,p0]),time) #numerical integration
    h=y[:,:n] #host frequencies
    p=y[:,n:] #parasite frequencies

#==============================================================================
# Plot Simplex
#==============================================================================

# Projection:
    proj=np.array([[-math.cos(30./360.*2.*math.pi), math.cos(30./360.*2.*math.pi),0.], 
        [-math.sin(30./360.*2.*math.pi),-math.sin(30./360.*2.*math.pi),1.]]) #projection Matrix (3D-->2D)

    [hx,hy] = np.array(np.mat(proj)*np.mat(h.T)) #2D plot values for host frequencies
    [px,py] = np.array(np.mat(proj)*np.mat(p.T)) #2D plot values for parasite frequencies
    if k==1:
        ax=axes[0]
    else:
        ax=axes[1]
    timeplot=ax.scatter(hx,hy,marker=".",alpha=0.05,c=time, cmap=cm, edgecolors="none",rasterized=True) #host plot

#    ax.scatter(hx[0],hy[0],marker='.',s=100,color="black",edgecolors="none") #initial condition host

    trianglepoints=np.hstack([np.identity(3),np.array([[1.],[0.],[0.]])]) #vertices
    triangleline=np.array(np.mat(proj)*np.mat(trianglepoints)) #edges
    ax.plot(triangleline[0],triangleline[1],clip_on=False,color="black",zorder=1) #plot edges for host simplex
# arrow to ini:
    ax.annotate(s="",xy=(hx[0],hy[0]),xycoords="data",xytext=(hx[0]-0.12,hy[0]+0.1),textcoords="data",arrowprops=dict(arrowstyle="->",shrinkA=0,shrinkB=0),ha="center",va="center")
# plot options:
    ax.annotate("$h_1$",xy=(0,0),xycoords="axes fraction",ha="right",va="top",fontsize=18,color="black")
    ax.annotate("$h_2$",xy=(1,0),xycoords="axes fraction",ha="left",va="top",fontsize=18,color="black")
    ax.annotate("$h_3$",xy=(0.5,1),xycoords="axes fraction",ha="center",va="bottom",fontsize=18,color="black")
#    ax.annotate("$h(0)=($"+"${:,.2f}$".format(h0[0])+'$,$ '+"${:,.2f}$".format(h0[1])+'$,$ '+"${:,.2f}$".format(h0[2])+'$)$',xy=(0.5,-0.1),xycoords="axes fraction", ha="center", va="top",fontsize=16)
    ax.set_xlim([triangleline[0,0],triangleline[0,1]]) #limit x-axis
    ax.set_ylim([-0.5,1.]) #limit y-axis
    ax.set_xlim([triangleline[0,0],triangleline[0,1]])
    ax.axis("off") #remove axis
    ax.set_aspect(1) #aspect ratio of 1
#axes[0].annotate("a",xy=(0,1),xycoords="axes fraction",fontsize=18,fontweight="bold")
#axes[1].annotate("b",xy=(0,1),xycoords="axes fraction",fontsize=18,fontweight="bold")
#fig.subplots_adjust(left=0.03,bottom=0.03,right=0.97,top=1.00)
fig.subplots_adjust(left=0.03,bottom=0.01,right=0.97,top=0.99)
#plt.savefig("simplex.pdf",rasterized=True,dpi=400)
plt.show()
