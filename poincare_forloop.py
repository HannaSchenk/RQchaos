# -*- coding: utf-8 -*-
"""
@author: schenk

Poincare sections of a host-parasite system with 3 types
Replicator dynamics
Matching allele model (see u,v,w)

the poincare sections are only visible if many steps and many generations are used
"""

from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import numpy as np

plt.close("all") # close figures
colours=['gold','limegreen','darkcyan','royalblue','indigo']  

#==============================================================================
# Variables and settings
#==============================================================================

ks=np.array([1,3,5,10,23]) #k in ]0,25] defines initial kondition
u=-1. #payoff matrix entry alpha1
v=0.  #payoff matrix entry alpha2
w=0.  #payoff matrix entry alpha3
generations=5000.
steps=50000. 
#generations=2.
#steps=2.

#==============================================================================
# Numerical integration and plot
#==============================================================================

mh=np.array([[u,v,w],[w,u,v],[v,w,u]]) #payoff matrix for host
#generations=2 #to test the program
#steps=10 #to test the program
n=3 #number of hosts/parasites
mp=-np.transpose(mh) #payoff matrix for parasite with c=1 
time = np.linspace(0.0,generations,steps) # start,end,steps 
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15.7,8.7)) #figure for poincare section    
axes[0].scatter([], [], color="white", marker='.',label="$h(0)$") #empty plotting for legend title

def deriv(y,t): #ode
    h=y[:n]
    p=y[n:]
    h=h/sum(h) #normalisation
    p=p/sum(p) #normalisation
    dH=np.dot(mh,p) #fitness of host influenced by parasite f_H
    bh=np.dot(dH,h) #avergae host fitness
    hdot=h*(dH-bh)  #ODE for hosts
    bP=np.dot(mp,h) #fitness of parasite influenced by host f_P
    dp=np.dot(bP,p) #average parasite fitness
    pdot=p*(bP-dp)  #ODE for parasites
    return np.concatenate([hdot,pdot]) #ODEs 

axesp=axes[0]
axesh=axes[1]

'''different initial conditions'''
counter=0
for k in ks: 
    h0=np.array([0.5,0.01*k,0.5-0.01*k])
    p0=np.array([0.5,0.25,0.25])
    y = odeint(deriv,np.concatenate([h0,p0]),time)
    h=y[:,:n]
    p=y[:,n:]
    axesp.scatter([], [], color=colours[counter], marker='.',label='$($'+'${:,.2f}$'.format(h0[0])+'$,$ '+'${:,.2f}$'.format(h0[1])+'$,$ '+'${:,.2f}$'.format(h0[2])+'$)$') #empty plotting for legend

#plot everything without condition:
#    axesp.scatter(h[:,0],p[:,1],marker='.',color=colours[counter],edgecolor='none',alpha=0.3)

#only plot condition:
    for t in np.arange(1,len(time),1):
        if -0.001<=(h[t,1]-h[t,0]+p[t,1]-p[t,0])<=0.001 and (h[t-1,1]-h[t-1,0]+p[t-1,1]-p[t-1,0])>=+0.001: 
            a='$h_2-h_1+p_2-p_1$'  
            axesp.scatter(h[t,0],p[t,1],marker='.',color=colours[counter],edgecolor='none')
        if -0.001<=np.log(h[t,0]*h[t,1]*h[t,2])-np.log(p[t,0]*p[t,1]*p[t,2])<=0.001 and np.log(h[t-1,0]*h[t-1,1]*h[t-1,2])-np.log(p[t-1,0]*p[t-1,1]*p[t-1,2])<-0.001: 
            a='$log(h_1*h_2*h_3)-log(p_1*p_2*p_3)$' # my constant of motion
            axesh.scatter(h[t,0],p[t,1],marker='.',color=colours[counter],edgecolor='none')
    counter+=1

'''plot options'''
axes[0].legend(prop={'size':16},scatterpoints=1,bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,markerscale=3) #set legend position
#axes[0].annotate("a",fontsize=18,fontweight="bold",xy=(-0.1,1.1),xycoords="axes fraction")
#axes[1].annotate("b",fontsize=18,fontweight="bold",xy=(-0.1,1.1),xycoords="axes fraction")
axes[0].set_xlabel('$h_1$',fontsize=18,family="sans-serif") #xlabel
axes[0].set_ylabel('$p_2$',fontsize=18,family="sans-serif") #ylabel
axes[0].set_xlim([0,0.8]) #xrange
axes[0].set_ylim([0,0.8]) #yrange
axes[0].set_aspect(1) #aspect ratio of plot
axes[0].annotate("$h_2-h_1+p_2-p_1=0$",ha="center",va="center",fontsize=18,xy=(0.5,1.1),xycoords="axes fraction")
axes[1].set_xlabel('$h_1$',fontsize=18,family="sans-serif")
axes[1].set_ylabel('$p_2$',fontsize=18,family="sans-serif")
axes[1].set_xlim([0,0.8])
axes[1].set_ylim([0,0.8])
axes[1].set_aspect(1)
axes[1].annotate("$log(h_1 h_2 h_3)-log(p_1 p_2 p_3)=0$",ha="center",va="center",fontsize=18,xy=(0.5,1.1),xycoords="axes fraction")
fig.subplots_adjust(left=0.04,bottom=0.08,right=0.99,top=0.88,wspace=0.62,hspace=0.04)
#fig.subplots_adjust(left=0.04,bottom=0.07,right=0.99,top=0.88,wspace=0.09)
fig.savefig("poincare.pdf",rasterized=True,dpi=1200)
plt.show()

