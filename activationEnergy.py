# Activation Energy
# Script is based on work
# S.P.M. Bane, J.L. Ziegler, and J.E. Shepherd.
# Development of One-Step Chemistry Models for Flame and Ignition Simulation
# link: shepherd.caltech.edu/EDL/publications/reprints/galcit_fm2010-002.pdf


from cantera import *
from SDToolbox import *
import math
from numpy import *
import scipy as Sci
import scipy.linalg
import numpy as np
import time
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def adiabaticTemp(T, P, mech, q, const):

    gas = Solution(mech);
    gas.TPX= T, P, q
    gas.equilibrate(const)

    return gas.T

def activationEnergy(mech, nC2H6, P0, Tinit, Tprim):
    ActEne = []
    C2H6perc = []
    R1 = 8.3144621 # J/mol*K
                     
    for i in range(0,len(nC2H6)):
        # nC2H6 = 0.1
        nO2 = (1-nC2H6[i])*0.21;
        nN2 = (1-nC2H6[i])*0.79;
        q = 'C2H6:'+str(nC2H6[i])+' O2:'+str(nO2)+' N2:'+str(nN2)+'';

        T0 = 0.9*adiabaticTemp(Tinit, P0, mech, q, "HP")

        t0 = cv_CJInd(0, P0, T0, q, mech, 0)
                 
        T1 = T0+Tprim
        
        t1 = cv_CJInd(0, P0, T1, q, mech, 0)
     
        AE = (R1*T0*((-T0*(t1-t0))/(t0*Tprim)+4))/1000 # kJ/mol

        ActEne.append(AE)
        C2H6perc.append(nC2H6[i])

    return ActEne

def cv_CJInd(plt_num, P1, T1, q, mech, fname):
    
    gas1 = Solution(mech);
    gas1.TPX= T1, P1, q
    r = gas1.density;
    [cj_speed,R2] = CJspeed(P1, T1, q, mech, plt_num)
    gas = PostShock_fr(cj_speed, P1, T1, q, mech);
    fname = 0;#fname + '_%d' % cj_speed
    #[exo_time,ind_time,ind_time_10,ind_time_90,time,temp,press,species] = explosion(gas);
    ind_time = explosionInd(gas);
    #return [cj_speed, gas]
    return ind_time


def explosionInd(gas):

    b = 1000; j = gas.n_species; rho = gas.density
    r = Reactor(gas)
    sim = ReactorNet([r])
    t = 0.0
    time = zeros(b,float)
    temp = zeros(b,float)
    press = zeros(b,float)
    temp_grad = zeros(b,float)
    y = zeros(j,float)
    species = zeros([b,j],float)

    #EXTRACT OUTPUT INFORMATION
    for n in range(b):
        time[n] = t
        temp[n] = r.T
        press[n] = r.thermo.P
        for i in range(j):
            species[n,i] = r.Y[i]
            y[i] = r.Y[i]

        gas.TRhoY= temp[n], r.density, y
        P = gas.P/one_atm

        # FIND TEMPERATURE GRADIENT
        # Conservation of Energy implies that e0 = e(T)
        # e = cv*T; dedt = 0; cv*dTdt + sum(ei(T)*dyidt) = 0; dyidt = wdoti*wti/rho
        # dTdt = -sum(ei(T)*wdoti*wti)/(cv*rho)
        cv = gas.cv_mass;
        wdot = gas.net_production_rates;
        mw = gas.molecular_weights
        hs = gas.standard_enthalpies_RT
        R = gas_constant;
        wt = gas.mean_molecular_weight

        sumT = 0.0
        for z in range(j):
            w = mw[z];
            e = R*temp[n]*(hs[z]/w - 1/wt)
            wd = wdot[z];
            sumT = sumT + e*wd*w;
        temp_grad[n] = -sumT/(rho*cv)
        t += 1.e-9
        sim.advance(t)

    del sim
    del r

    #FIND INDUCTION TIME - MAXIMUM TEMPERATURE GRADIENT
    k = 0; MAX = max(temp_grad); d = temp_grad[0]; HMPWt = zeros(2,float)
    if d == MAX:
        print 'Initial Gradient is Maximum - post shock temperature may be too low'
        return gas
    while d < MAX:
        k = k + 1; d = temp_grad[k];
    ind_time = time[k]; k1 = k; k = 0;
    MAX10 = 0.1*MAX; d = temp_grad[0];
    while(d < MAX10 and k < b-1):
        k = k + 1; d = temp_grad[k];
    if(k == b):
        print 'MAX10 may be incorrect - reached end of array'
    ind_time_10 = time[k]; k = 0;
    MAX90 = 0.9*MAX; d = temp_grad[0];
    while(d < MAX90 and k < b-1):
        k = k + 1; d = temp_grad[k];
    if(k == b-1):
        print 'MAX90 may be incorrect - reached end of array'
    ind_time_90 = time[k];

    return ind_time


#############################################
###mech = 'h2air_highT.cti'
###mech = 'h2o2mech_Lietal_2003.cti'
#mech = 'gri30.cti'
mech = 'gri30_highT.cti'

#nC2H6 = [0.03, 0.035875, 0.04175, 0.047625, 0.0535, 0.059375, 0.06525, 0.071125, 0.077, 0.082875, 0.08875, 0.094625, 0.1005, 0.106375, 0.11225, 0.118125, 0.124]
nC2H6 = [x * 0.001 for x in range(30, 127, 7)] # about 3 - 12.4 percentage concentration
#R1 = 1.9872041347992353 # cal/mol*K

T1fin = []
P0 = one_atm
Tinit = 300
Tprim = 30

AE = activationEnergy(mech, nC2H6, P0, Tinit, Tprim)

ActEnePfit1 = np.poly1d(np.polyfit(nC2H6, AE, 5))   # 
ActEnePfit = np.polyfit(nC2H6, AE,  5)              # finds polynomial connecting the points (x,y, degree)



x = nC2H6
ActEneLine = []
for i in x:
    #ActEneLine.append(ActEnePfit[0]*i**6+ActEnePfit[1]*i**5+ActEnePfit[2]*i**4+ActEnePfit[3]*i**3+ActEnePfit[4]*i**2+ActEnePfit[5]*i+ActEnePfit[6])
    ActEneLine.append(ActEnePfit[0]*i**5+ActEnePfit[1]*i**4+ActEnePfit[2]*i**3+ActEnePfit[3]*i**2+ActEnePfit[4]*i+ActEnePfit[5])
    #ActEneLine.append(ActEnePfit[0]*i**2+ActEnePfit[1]*i**1+ActEnePfit[2])
    


plt.plot(nC2H6, AE,'-', label="ActivEner")
plt.plot(x, ActEneLine, '--', label="ActEneLine")
plt.plot(nC2H6, ActEnePfit1(nC2H6),'ro', label="ActEnePfit1")
plt.xlabel("C2H6 concentration [%]")
plt.ylabel("Activation energy [kJ/mol]")
#plt.ylim([0,100])
plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator(n=4))
plt.gca().yaxis.set_minor_locator(tck.AutoMinorLocator(n=4))
plt.grid(which="both")
plt.legend(loc="upper right")
plt.show()

with open('activation_energy.txt','w') as sfile:
    sfile.write("C2H6 %" + "\t" + "Act.en." + "\t" + "Act.en.fit." + "\n")
    for n in range(len(x)):
        sfile.write(str(x[n]) + "\t" + str(AE[n]) + "\t" + str(ActEneLine[n]) + "\n")

"""
for i in range(0,len(nC2H6)):   
    nO2 = (1-nC2H6[i])*0.21;
    nN2 = (1-nC2H6[i])*0.79;
    q = 'C2H6:'+str(nC2H6[i])+' O2:'+str(nO2)+' N2:'+str(nN2)+'';
""" 
