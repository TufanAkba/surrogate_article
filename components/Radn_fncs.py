#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import pi, sin, cos, sqrt, asin, log
from numpy.random import random_sample as rand
# import time

def calc_F(r_1,L,disc_z,dL):
    "Cyclindircal to cylindirical element and BP"

    F_temp = np.empty((disc_z,1),dtype=float)
    F_1_BP = np.empty((disc_z,1),dtype=float)
    
    
    F_temp[0,0] = (1+dL/(2*r_1))-(1+(dL/(2*r_1))**2)**(0.5);
    
    L1 = dL/r_1
    for i in range(1,disc_z):
        
        L2=(dL+(i-1)*dL)/r_1;
        L3=(2*dL+(i-1)*dL)/r_1;
        X1=((L3-L1)**2+4)**0.5;
        X2=((L2-L1)**2+4)**0.5;
        X3=((L3)**2+4)**0.5;
        X4=((L2)**2+4)**0.5;
        F_temp[i,0]=1/(4*(L3-L2))*(2*L1*(L3-L2)+(L3-L1)*X1-(L2-L1)*X2-L3*X3+L2*X4);

        H2=(i-1)*L1;
        F_1_BP[disc_z-i,0]=0.25*((1+H2/L1)*(4+(L1+H2)**2)**0.5-(L1+2*H2)-H2/L1*(4+H2**2)**0.5);
    
    H2 = (disc_z-1)*L1
    F_1_BP[0,0]=0.25*((1+H2/L1)*(4+(L1+H2)**2)**0.5-(L1+2*H2)-H2/L1*(4+H2**2)**0.5);
    
    return F_temp,F_1_BP



def MoCa_3D(disc_z, r_1, L, dL):
    
    file = 'radn_data/'+str(disc_z)+str(r_1)+str(L)
    try:
        # F_I_BP = np.load('radn_data/'+str(disc_z)+str(r_1)+str(L)+'F_I_BP'+'.npy')
        # F_I_1 = np.load('radn_data/'+str(disc_z)+str(r_1)+str(L)+'F_I_1'+'.npy')
        F_I_BP = np.load(file+'F_I_BP'+'.npy')
        F_I_1 = np.load(file+'F_I_1'+'.npy')
        return F_I_1,F_I_BP
    except IOError:
        ConeAngle=45; #angle over which the entering radiation is distributed
        rays=500000; #number of rays
        n_BP=0
    
        n=np.zeros((1,disc_z))
        ConeAngle=ConeAngle*pi/180;
        # r_1_temp = r_1
        r_1 = r_1**2 #r_1 Only used in squares
    
        for i in range(rays):
            #random location
            fi=2.0*pi*rand()                #eqn 5.10
            r=sqrt(rand()*r_1);             #eqn 5.9
            x=r*cos(fi);                    #coordinates
            y=r*sin(fi);
            #z=0;
            #random direction
            theta=asin(sin(ConeAngle)*sqrt(rand()))   #eqn 5.11
            fi=2*pi*rand();                           #eqn 5.12
            ux=sin(theta)*cos(fi);
            uy=sin(theta)*sin(fi);
            uz=cos(theta);
    
            lam=L/uz;       #stretch factor
            X=x+ux*lam;     #end point
            Y=y+uy*lam;
    
            #Does it hit to back plate?
            if ((X**2+Y**2)<=r_1):
                n_BP = n_BP + 1
            else:
                lam=(sqrt((ux**2+uy**2)*r_1-x**2*uy**2+2*x*y*ux*uy-y**2*ux**2)-x*ux-y*uy)/(ux**2+uy**2);
                # Z=lam*uz;       #where does ray hit in the cylinder jacket
                Z_mesh = int((lam*uz)//dL)
                n[(0,Z_mesh)] = n[(0,Z_mesh)] + 1
    
        F_I_BP=n_BP/rays;
        F_I_1=n/rays;
        
        # np.save('radn_data/'+str(disc_z)+str(r_1_temp)+str(L)+'F_I_BP',F_I_BP)
        # np.save('radn_data/'+str(disc_z)+str(r_1_temp)+str(L)+'F_I_1',F_I_1)
        np.save(file+'F_I_BP',F_I_BP)
        np.save(file+'F_I_1',F_I_1)
        
        return F_I_1,F_I_BP
        
    
    


def I_Rad(T_RPC,r,K_ex,omega,E1,E2,G_Initial,residual):
        
        sigma=5.67*(10**(-8));
        G_0 = np.copy(G_Initial);
        G_1 = np.copy(G_0);
        g_residual=100;
        disc_r = T_RPC.shape[0]-2
            
        while g_residual>0.1*abs(residual):
            # Boundary Condition
            K1=-2/3*(2-E1)/E1*1/K_ex;
            K2=-2/3*(2-E2)/E2*1/K_ex;
            K3=4*sigma*T_RPC[0]**4;
            K4=4*sigma*T_RPC[-1]**4;
            G_1[0]=(K3*r[0]*log(r[1]/r[0])-G_0[1]*K1)/(r[0]*log(r[1]/r[0])-K1);
            G_1[-1]=(K4*r[-1]*log(r[-1]/r[-2])-G_0[-2]*K2)/(r[-1]*log(r[-1]/r[-2])-K2);
            
            # Internal Nodes
            for i in range(2,disc_r+2):
                # print(i)
                C_A=1/log(r[i-1]/r[i-2]);
                C_B=1/log(r[i]/r[i-1]);
                C_C=-C_A-C_B-r[i-1]*log((r[i-1]+r[i])/(r[i-1]+r[i-2]))*3*r[i-1]*K_ex**2*(1-omega);
                C_D=-r[i-1]*log((r[i-1]+r[i])/(r[i-1]+r[i-2]))*3*r[i-1]*K_ex**2*(1-omega)*4*sigma*T_RPC[i-1]**4;
                G_1[i-1]=(C_D-C_A*G_0[i-2]-C_B*G_0[i])/C_C;
                
            g_residual=max(np.true_divide((G_1-G_0),G_1));
            G_0=np.copy(G_1);
        return G_0