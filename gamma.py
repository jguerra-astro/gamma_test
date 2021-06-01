import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib
matplotlib.rcParams.update({'font.size': 30})


def rho(r,theta):
    rho_d,r_d,a,b,c,beta= theta
    rho_d = np.exp(rho_d)* u.solMass/(u.kpc**3)
    r_d   = np.exp(r_d)  * u.kpc
    x   = r/r_d
    nu  = (a-c)/b
    out = rho_d*x**(-a) * (1+x**b)**nu
    return out

def gamma(r,theta):
    rho_d,r_d,a,b,c,beta= theta
    rho_d = np.exp(rho_d)* u.solMass/(u.kpc**3)
    r_d   = np.exp(r_d)  * u.kpc
    x   = r/r_d
    nu  = (a-c)/b
    gamma = a - (a-c)*(r/r_d)**b/(1.0+(r/r_d)**b)
    return gamma

def func(x):
    return ("%4.2f"% (x))


theta_cusp = (np.log(2e7),np.log(2),1,1,3,0)
theta_core = (np.log(3e8),np.log(.5),0,1.5,3,0)
r_d = np.logspace(-1.5,np.log10(3),100)
# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

# y = st.sidebar.slider('Select a range of values for r_d : ',.001, 3.0,1.0,step=.05)



r          = np.logspace(-3,1,40) * u.kpc

theta_cusp = (np.log(2e7),np.log(2),1,1,3,0)
theta_core = (np.log(3e8),np.log(.5),0,1.5,3,0)
rho_cusp   = rho(r,theta_cusp)
rho_core   = rho(r,theta_core)
gamma_cusp = gamma(r,theta_cusp)
gamma_core = gamma(r,theta_core)
fig,ax     = plt.subplots(figsize=(25,10),ncols=2)
# ax[0].plot(r,rho_cusp,lw=3+1)
# ax[0].plot(r,rho_core,lw=4+1,color='black')
# ax[0].plot(r,rho_core,lw=3+1)
ax[0].set(xscale='log',yscale='log',xlabel='R [kpc]',ylabel=r'$\rho_d$')
r_s = np.logspace(-1.5,.3,15)
for i in r_s:
    theta = (np.log(3e8),np.log(i),0,1.5,3,0)
    rhod   = rho(r,theta)
    ax[0].plot(r,rhod,lw=3,color="b", alpha=0.1)
    ax[0].scatter(i,rho(i*u.kpc,theta),color="b", alpha=0.1)
for i in r_s:
    theta = (np.log(2e7),np.log(i),1,1,3,0)
    rhod   = rho(r,theta)
    ax[0].plot(r,rhod,lw=3,color="r", alpha=0.1)
    ax[0].scatter(i,rho(i*u.kpc,theta),color="r", alpha=0.1)
############################################################################
# ax[1].plot(r,gamma_cusp,lw=3+1)
# ax[1].plot(r,gamma_core,lw=4+1,color='black')
# ax[1].plot(r,gamma_core,lw=3+1)
ax[1].axvline(.25,color='black')
for i in r_s:
    theta = (np.log(3e8),np.log(i),0,1.5,3,0)
    rhod   = gamma(r,theta)
    ax[1].plot(r,rhod,lw=3,color="b", alpha=0.1)
    ax[1].scatter(i,gamma(i*u.kpc,theta),color="b", alpha=0.1)
for i in r_s:
    theta = (np.log(2e7),np.log(i),1,1,3,0)
    rhod   = gamma(r,theta)
    ax[1].plot(r,rhod,lw=3,color="r", alpha=0.1)
    ax[1].scatter(i,gamma(i*u.kpc,theta),color="r", alpha=0.1)
    
ax[1].set(xscale='log',xlabel='R [kpc]',ylabel ='$log-slope$')

x = st.sidebar.select_slider('Cusp: Select a range of values for r_d :',r_d,format_func=func)

y = st.sidebar.select_slider('Core Select a range of values for r_d :',r_d,format_func=func)

ax[0].plot(r,rho(r,(np.log(2e7),np.log(x),1,1,3,0)),lw=5,color='red',label='cusp')
ax[1].plot(r,gamma(r,(np.log(2e7),np.log(x),1,1,3,0)),lw=5,color='red')
ax[0].plot(r,rho(r,(np.log(3e8),np.log(y),0,1.5,3,0)),lw=5,color='blue',label='core')
ax[1].plot(r,gamma(r,(np.log(3e8),np.log(y),0,1.5,3,0)),lw=5,color='blue')
ax[0].legend(fontsize=30)
st.write(fig)

# st.write(y, 'squared is', y * y)











