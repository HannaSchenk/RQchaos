########################################
#    Copyright 2017 Hanna Schenk
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################

"""

Plot time trajectory, 3-simplex and Poincare section for bimatrix replicator 
dynamics with two species (host, parasite), three types each or Lotka Volterra 
dynamics with two species, two types each.

"""
from scipy.integrate import odeint 
from scipy.integrate import ode
import matplotlib.pyplot as plt
import numpy as np
import math
import odespy


#==============================================================================
# Main settings
#==============================================================================
integrator = 'lsodar'  # choose from lsoda (odeint), vode, dopri5, dop853 (int) and lsodar for a root finding from odespy
log_integration = 0

plot_trajectory = 1  # choose whether to plot trajectory (1) or not (0)
trajectory_k = 3  # choose which initial condition to use for trajectory (out of ks)\

plot_simplex = 1  # choose whether to plot simplex or 2D plot (1) or not (0)
simplex_k = [1,10]  # choose two ks for simplex
ini_dot = 1  # show initial condition as dot
ini_arrow = 0  # show initial condition as arrow

plot_poincare = 1  # choose whether to plot poincare section (1) or not (0) 

testing = 0  # testing - less time steps

RD = 1  # chose replicator dynamics (1) or Lotka-Volterra (0)

#==============================================================================
# More variables and settings
#==============================================================================
if plot_poincare == 1 and integrator!= 'lsodar':
    exit('please choose different integrator if Poincare section should be calculated')
if plot_poincare == 0 and integrator == 'lsodar':
    exit('please do not choose lsodar if you do not wish to calculate the Poincare section')

if RD == 1:  # replicator dynamics
    ## matching allele interactions:
    a = -1.
    b = 0.
    e = 0.
    mh = np.array([[a,b,e],[e,a,b],[b,e,a]])  # payoff matrix for host

c = 1.  # factor for payoff matrices

if RD == 0:  # Lotka volterra dynamics
    a = -1.
    b = -0.02  # 'alpha_2' coupling of equations
    mh = np.array([[a,b],[b,a]])  # payoff matrix for host
    Bh = 5.  # constant birthrate for host
    Dp = 2.5  # constant deathrate for parasites

ks = [1,3,5,10,23]  # initial condition
if plot_simplex == 0 and plot_poincare == 0:
    ks = [trajectory_k]
colours = ['gold','limegreen','darkcyan','royalblue','indigo']  # colours for different initial conditions
cm_viridis = plt.cm.get_cmap('viridis')

## 3 figures
if plot_trajectory == 1:
    fig_trajectory, axes_trajectory = plt.subplots(nrows=1,ncols=1,figsize=(15.7,7.7)) 
if plot_simplex == 1:
    fig_simplex, axes_simplex = plt.subplots(nrows=1,ncols=2,figsize=(15.7,7.7)) 
if plot_poincare == 1:
    fig_poincare, axes_poincare = plt.subplots(nrows = 1, ncols = 2, figsize = (15.7, 7.7))
#==============================================================================
# Condition for poincare plot
#==============================================================================

def g(y,t):  # root conditions
    h = y[:n]
    p = y[n:]
    if RD == 1:  # two planes for replicator dynamics
        return [h[1] - h[0] + p[1] - p[0], np.log(h[0] * h[1] * h[2]) - np.log(p[0] * p[1] * p[2])]
    if RD == 0:  # two planes for Lotka-Volterra dynamics
        return [h[0] + h[1] - 0.6, h[0] + h[1] - p[0] - p[1]] 

#==============================================================================
# Define functions for ODEs
#==============================================================================

def deriv(y,t):  # define ODEs
    if log_integration == 1:
        y = np.exp(y)
    h = y[:n]  # host frequencies
    p = y[n:]  # parasite frequencies
    if RD == 1:  # Replicator dynamics
        h = h/sum(h)  # normalisation
        p = p/sum(p)  # normalisation
    dH = np.dot(mh, p) #host fitness influenced by parasites
    bP = np.dot(mp, h)  # parasite fitness influenced by host
    if RD == 1:  # Replicator dynamics
        bh = np.dot(dH, h)  # average host fitness
        dp = np.dot(bP, p)  # aberage parasite fitness
        hdot = h * (dH - bh)   # ODE for hosts
        pdot = p * (bP - dp)  # ODE for parasites         
    if RD == 0:  # Lotka-Volterra
        bh = Bh  # constant birthrate for host
        dp = Dp  # constant deathrate for parasite
        hdot = h * (dH + bh)   # ODE for hosts
        pdot = p * (bP - dp)  # ODE for parasites
    if log_integration == 1:
        hdot = np.log(hdot)
        pdot = np.log(pdot)
    return np.concatenate([hdot,pdot])  # all ODEs

#==============================================================================
# Loop through initial conditions
#==============================================================================   
for index_k,k in enumerate(ks):  # for each initial condition
    print("initial condition k: ",k)
    if plot_poincare == 1:
        generations = 100000.  # time steps to integrate over
        steps=10000000  # actual number of outputs (points to plot) - integration steps
    if plot_poincare == 0 and plot_simplex == 1:
        generations = 10000.
        steps = 1000000
    if plot_poincare == 0 and plot_simplex == 0:
        generations = 300.
        steps = 30000
    if testing == 1:  # few generations to test program
        generations = 300.
        generations = 200.
        steps = 3000

    step = generations/steps  # stepsize for integrator (not internal)
    time = np.linspace(0.01,generations,steps)  # start,end,steps 
    if RD == 1:  # replicator dynamics
        h0 = np.array([0.5, 0.01 * k, 0.5 - 0.01 * k])  # initial host frequencies
        p0 = np.array([0.5, 0.25, 0.25])  # initial parasite frequencies
    if RD == 0:  # Lotka-Volterra dynamics
        h0 = np.array([0.5,0.01 * k])  # initial host frequencies
        p0 = np.array([0.5,0.25])  # initial parasite frequencies

    ## log integration
    if log_integration == 1:
        time = np.log(time)
        h0 = np.log(h0)
        p0 = np.log(p0)

    n = len(h0)  # number of hosts or parasites
    mp = -c * mh.T  # payoff matrix for parasites

    #==============================================================================
    # Solving of ODE for each initial condition
    #==============================================================================


    if integrator == 'lsoda':  # lsoda --> use odeint
            y = odeint(deriv,np.concatenate([h0,p0]),time)  # numerical integration with lsoda

    elif integrator == 'lsodar':  # lsodar --> use odespy
            solver = odespy.Lsodar(deriv, g = g, atol = 10**(-9))  # atol - precision
            solver.set_initial_condition(np.concatenate([h0,p0]))  # set initial conditions as above
            y, time, roots, roots_time = solver.solve(time)  # solve
            roots = np.array(roots)  # location of roots for poincare section

    else:  # neither lsoda nor lsodar --> use ode.integrate

        def deriv2(t,y):  # define ODEs
            return deriv(y,t)  # switch y and t for this integrator
     
        t = 0  # start time
        r = ode(deriv2).set_integrator(integrator, atol = 10**(-30), nsteps=10**6, max_step = 0.01, dfactor = 10**5)
        r.set_initial_value(np.concatenate([h0, p0]), t)  # initial conditions as above
        tmax = generations  # maxmum time
        dt = step  # stepsize
        y = np.concatenate([h0, p0]).reshape(1,6)  # initiate output with initial consition
        while r.successful() and r.t < tmax:  # while tmax is not reached
            r.integrate(r.t + dt)  # integrate
            t = np.append(t, r.t)  # append time
            y = np.append(y, r.y.reshape(1,6), axis = 0)  # append output
    
    if log_integration == 1:
        y = np.exp(y)  # frequencies
        time = np.exp(time)
    y = y[:len(time),:]
    h = y[:,:n]  # host frequencies
    p = y[:,n:]  # parasite frequencies

    #==============================================================================
    # Plot Trajectory
    #==============================================================================

    if plot_trajectory == 1 and k==trajectory_k:
        colours_trajectory = ["lightskyblue","dodgerblue","navy"]
        tmax_trajectory = 300-1
        time_trajectory = time[time<tmax_trajectory]
        ## plot h_1 and h_2
        h1, =axes_trajectory.plot(time_trajectory,h[time<tmax_trajectory,0],color=colours_trajectory[0],label=r'$h_1$')
        h2, =axes_trajectory.plot(time_trajectory,h[time<tmax_trajectory,1],color=colours_trajectory[1],label=r'$h_2$')
        if RD == 1:  # plot h_3 for replicator dynamics
            h3, =axes_trajectory.plot(time_trajectory,h[time<tmax_trajectory,2],color=colours_trajectory[2],label=r'$h_3$')
        ## plot p_1 and p_2
        p1, =axes_trajectory.plot(time_trajectory,p[time<tmax_trajectory,0],color=colours_trajectory[0],linestyle='--',label=r'$p_1$')
        p2, =axes_trajectory.plot(time_trajectory,p[time<tmax_trajectory,1],color=colours_trajectory[1],linestyle='--',label=r'$p_2$')
        if RD == 1:  # plot p_3 for replicator dynamics
            p3, =axes_trajectory.plot(time_trajectory,p[time<tmax_trajectory,2],color=colours_trajectory[2],linestyle='--',label=r'$p_3$')
    #==============================================================================
    # Plot Simplex
    #==============================================================================

    if plot_simplex==1 and (k==simplex_k[0] or k==simplex_k[1]):
        # print("yes")
        tmax_simplex = 5000 if k>4 else 10000
        if RD == 0:
            tmax_simplex = int(generations)
        index_k_simplex = 0 if k==simplex_k[0] else 1
        h_simplex = h[time<=tmax_simplex,:]
        p_simplex = p[time<=tmax_simplex,:]
        ax = axes_simplex[index_k_simplex]
        if RD == 1:
            # Projection:
            proj=np.array([[-math.cos(30./360.*2.*math.pi), math.cos(30./360.*2.*math.pi),0.], 
                [-math.sin(30./360.*2.*math.pi),-math.sin(30./360.*2.*math.pi),1.]]) #projection Matrix (3D-->2D)
            [hx,hy] = np.array(np.mat(proj)*np.mat(h_simplex.T)) #2D plot values for host frequencies
            [px,py] = np.array(np.mat(proj)*np.mat(p_simplex.T)) #2D plot values for parasite frequencies
            timeplot = ax.scatter(hx,hy,marker=".",alpha=0.05,c=time[time<=tmax_simplex], cmap=cm_viridis, edgecolors="none",rasterized=True) #host plot
            trianglepoints=np.hstack([np.identity(3),np.array([[1.],[0.],[0.]])]) #vertices
            triangleline=np.array(np.mat(proj)*np.mat(trianglepoints)) #edges
            ax.plot(triangleline[0],triangleline[1],clip_on=False,color="black",zorder=1) #plot edges for host simplex
        if RD ==0:  # no projection for Lotka-Volterra
            if index_k_simplex == 0:
                timeplot = ax.scatter(h_simplex[:,0],p_simplex[:,0],marker=".",alpha=0.05,c=time[time<tmax_simplex], cmap=cm, edgecolors="none",rasterized=True) #host plot
                ax.set_xlabel(r'$h_1$', fontsize = 18)
                ax.set_ylabel(r'$p_1$', fontsize = 18)
            if index_k_simplex == 1:
                timeplot = ax.scatter(h_simplex[:,0],h_simplex[:,1],marker=".",alpha=0.05,c=time[time<tmax_simplex], cmap=cm, edgecolors="none",rasterized=True) #host plot
                ax.set_xlabel(r'$h_1$', fontsize = 18)
                ax.set_ylabel(r'$h_2$', fontsize = 18)
    ###### arrow to initial conditions ######
        if ini_arrow == 1 and RD == 1:
            ax.annotate(s="",xy=(hx[0],hy[0]),xycoords="data",xytext=(hx[0]-0.12,hy[0]+0.1),textcoords="data",arrowprops=dict(arrowstyle="->",shrinkA=0,shrinkB=0),ha="center",va="center")
    ###### point at initial conditions ######
        if ini_dot ==1 and RD == 1:
            ax.scatter(hx[0],hy[0],marker='.',s=100,color="black",edgecolors="none") #initial condition host

        if RD == 1:
        ###### corner labels ######
            ax.annotate("$h_1$",xy=(0,0),xycoords="axes fraction",ha="right",va="top",fontsize=18,color="black")
            ax.annotate("$h_2$",xy=(1,0),xycoords="axes fraction",ha="left",va="top",fontsize=18,color="black")
            ax.annotate("$h_3$",xy=(0.5,1),xycoords="axes fraction",ha="center",va="bottom",fontsize=18,color="black")

        ###### initial conditions ######
            ax.annotate("$h(0)=($"+"${:,.2f}$".format(h0[0])+'$,$ '+"${:,.2f}$".format(h0[1])+'$,$ '+"${:,.2f}$".format(h0[2])+'$)$',xy=(0.5,-0.1),xycoords="axes fraction", ha="center", va="top",fontsize=16)

            ax.set_xlim([triangleline[0,0],triangleline[0,1]]) #limit x-axis
            ax.set_ylim([-0.5,1.]) #limit y-axis
            ax.set_xlim([triangleline[0,0],triangleline[0,1]])
            ax.axis("off") #remove axis
            ax.set_aspect(1) #aspect ratio of 1
            ax.set_xlabel(r'$k={}$'.format(k))
        if RD == 0:
            ax.annotate(r'$k={}$'.format(k), ha="center", va="center", fontsize=18, xy=(0.5,1.05), xycoords="axes fraction")

    #==============================================================================
    # Plot Poincare
    #==============================================================================
    if plot_poincare == 1:
        if index_k == 0:
            axes_poincare[0].scatter([], [], color="white", marker='.',label="$h(0)$") #empty plotting for legend title
        for i in [0,1]:  # two conditions
            current_roots = np.array(roots[i])  # roots for condition i
            print("shape current_roots", np.shape(current_roots))  # print how many roots are found
            if np.shape(current_roots)[0]>0:
                axes_poincare[i].scatter(current_roots[:,0], current_roots[:,n+1], alpha = 0.3, marker = '.', color = colours[index_k],edgecolor='none',rasterized=True) #plot
        if RD == 1:
            axes_poincare[0].scatter([], [], color=colours[index_k], marker='.',label='$($'+'${:,.2f}$'.format(h0[0])+'$,$ '+'${:,.2f}$'.format(h0[1])+'$,$ '+'${:,.2f}$'.format(h0[2])+'$)$') #empty plotting for legend
        if RD == 0:
            axes_poincare[0].scatter([], [], color=colours[index_k], marker='.',label='$($'+'${:,.2f}$'.format(h0[0])+'$,$ '+'${:,.2f}$'.format(h0[1])+'$)$') #empty plotting for legend

#==============================================================================
# Plot Options
#==============================================================================

if plot_trajectory == 1:  # plot for trajectory (fig 1)
    if RD == 1:
        axes_trajectory.set_ylim([0,1])
        axes_trajectory.set_aspect(300/2.5)
    axes_trajectory.legend()
    axes_trajectory.set_ylabel(r'relative abundance',fontsize=18)
    axes_trajectory.set_xlabel('time',fontsize=18)

if plot_simplex == 1:  # plot for simplex (fig 2)
    if RD == 1:
        fig_simplex.subplots_adjust(left=0.03,bottom=0.01,right=0.97,top=0.99)  # positioning of plots  
        axes_simplex[0].annotate("a",xy=(0,1),xycoords="axes fraction",fontsize=18,fontweight="bold")  # subfigure a
        axes_simplex[1].annotate("b",xy=(0,1),xycoords="axes fraction",fontsize=18,fontweight="bold")  # subfigure b
    if RD == 0:
        fig_simplex.subplots_adjust(left=0.05,bottom=0.1,right=0.9,top=0.9)  # positioning of plots
        axes_simplex[0].annotate("a",xy = (-0.01,1.03),xycoords="axes fraction",fontsize=18,fontweight="bold")  # subfigure a
        axes_simplex[1].annotate("b",xy = (-0.01,1.03),xycoords="axes fraction",fontsize=18,fontweight="bold")  # subfigure b

if plot_poincare == 1:
    if RD == 1:  # replicator dynamics
        cond = ["$h_2-h_1+p_2-p_1=0$","$ log ( h_1  h_2  h_3 ) - log (p_1 p_2 p_3)=0$"]  # planes for Poincare section
    if RD == 0:  # Lotka-Volterra
        cond = ["$ h_1 + h_2 = 0.6$", "$h_1 + h_2 - p_1 - p_2= 0$"]  # planes for Poincare section
    for i in [0,1]:  # loop through subfigures
        axes_poincare[i].set_xlabel('$h_1$', fontsize=18,family="sans-serif")  # xlabel
        axes_poincare[i].set_ylabel('$p_2$',fontsize=18,family="sans-serif")  # ylabel
        if RD == 1:  # replicator dynamics
            axes_poincare[i].set_xlim([0,0.8])  # xrange
            axes_poincare[i].set_ylim([0,0.8])  # yrange
            axes_poincare[i].set_aspect(1)  # aspect ratio of plot
        if RD == 0:  # Lotka-Volterra
            axes_poincare[i].set_xlim(left=0)  # lowest xvalue
            axes_poincare[i].set_ylim(bottom=0)  # lowest yvalue
        axes_poincare[i].annotate(cond[i],ha="center",va="center",fontsize=18,xy=(0.5,1.05),xycoords="axes fraction")  # write condition for plane
    axes_poincare[0].legend(prop={'size':13},scatterpoints=1,bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,markerscale=3)  # legend
    if RD == 1:  # replicator dynamics
        fig_poincare.subplots_adjust(left=0.06,bottom=0.08,right=0.99,top=0.85,wspace=0.5,hspace=0.04)  # positioning of plots
    if RD == 0:  # Lotka-Volterra
        fig_poincare.subplots_adjust(left=0.06,bottom=0.08,right=0.99,top=0.85,wspace=0.45,hspace=0.04)  # positioning of plots
    axes_poincare[0].annotate("a",fontsize=18,fontweight="bold",xy=(-0.1,1.1),xycoords="axes fraction")  # subfigure a
    axes_poincare[1].annotate("b",fontsize=18,fontweight="bold",xy=(-0.1,1.1),xycoords="axes fraction")  # subfigure b

plt.show()  # show all figures
