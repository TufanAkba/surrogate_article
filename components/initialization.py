#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:51:14 2021

@author: tufan

Promotes all inputs and outputs,
Basic calculator for initial parameters
No iteration exists or solver required
"""

import openmdao.api as om
from math import pi, sqrt
import numpy as np
from components.Radn_fncs import MoCa_3D
import time

class initialization(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        

        
        self.porosity = 0.81
        self.p = 10
        self.D_nom = 0.00254
        
        self.d0=1.5590;
        self.d1=0.5954;
        self.d2=0.5626;
        self.d3=0.4720;
        
        # print('initialization.initialize')
        
    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        self.disc_r=disc_SiC+disc_RPC+disc_INS;
        self.Nr=self.disc_r+4;
        self.Nz=disc_z+2;
        self.NN=self.Nz*self.Nr;
        
        self.bound1=disc_SiC+2;
        self.bound2=disc_SiC+2+disc_RPC+1;
        
        self.F_1_BP = np.empty((1,disc_z),dtype=float)
        
        self.add_input('L',val=0.065,desc='Length of the SiC tube', units='m')
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('s_SiC',val=0.005,desc='Thikness of SiC tube', units='m')
        self.add_input('s_RPC',val=0.015,desc='Thikness of RPC tube', units='m')
        self.add_input('s_INS',val=0.1,desc='Thickness Insulation', units='m')
        
        self.add_input('h_loss',val=15,desc='Heat transfer coefficient to ambient',units='W/(m**2*K)')
        
        self.add_input('k_INS',val=0.3,desc='Conductivity insulation',units='W/(m*K)')
        self.add_input('k_SiC',val=33,desc='Conductivity SiC',units='W/(m*K)')
        self.add_input('k_Air',val=0.08,desc='Conductivity air',units='W/(m*K)')
        
        # self.add_input('porosity',val=0.81,desc='Porosity RPC')
        
        # self.add_input('p',val=10,desc='Pressure',units='bar')
        
        mass_Flow = self.add_input('Mass_Flow', val=0.00068,desc='Mass flow rate', units='kg/s')
        
        # self.add_input('D_nom',val=0.00254,desc='Nominal pore size', units='m')
        
        # self.add_input('Tamb',val=293,desc='Ambient Temperature', units='K')
        
        self.add_input('E',val=0.9,desc='Emissivity')
        
        self.add_output('dL', desc='Axial mesh length', units='m')
        
        # for radiocity
        self.add_output('B', shape=(disc_z+1,disc_z+1), desc='Coeff. matrix of the radiocity')
        
        self.add_output('F', shape=(disc_z,disc_z), desc='View factor cavity to cavity')
        self.add_output('F_I_1', shape=(1,disc_z), desc='View factor aperture to cavity')
        self.add_output('F_I_BP', desc='View factor aperture to BP')
        self.add_output('F_1_BP', shape=(1,disc_z), desc='View factor cavity to BP')
        self.add_output('F_BP_1', shape=(1,disc_z), desc='View factor BP to cavity')
        
        # for fluid
        self.add_output('h_',desc='Heat transfer coefficient inside RPC',units='W/(m**2*K)')
        self.add_output('V_RPC', shape=(disc_RPC,1), desc='Volume of each RPC element', units='m**3')
        self.add_output('m', shape=(disc_RPC,1), desc='mass flow rate passing inside each RPC element', units='kg/s')
        
        #for solid
        self.add_output('r_n', shape=(self.Nr,1), units='m', desc='radial node coordinates')
        self.add_output('r_g', shape=(self.Nr-3,1), units='m', desc='radial grid coordinates')
        self.add_output('z_n', shape=(1,disc_z+2), units='m', desc='axial grid coordinates')
        
        self.add_output('h_loss_cav', units='W/(m**2*K)', desc='heat transfer coefficient from cavity')
        self.add_output('h_loss_z', units='W/(m**2*K)', desc='heat transfer coefficient on vertical surfaces')
        
        self.add_output('k_RPC', desc='Conductivity RPC',units='W/(m*K)')
        
        self.add_output('Ac_SiC', shape=(disc_SiC,1), units='m**2', desc='Cross sectional area of SiC elements')
        self.add_output('Ac_RPC', shape=(disc_RPC,1), units='m**2', desc='Cross sectional area of RPC elements')
        self.add_output('Ac_INS', shape=(disc_INS,1), units='m**2', desc='Cross sectional area of INS elements')
        self.add_output('Volume', val=0.0037216091972588094, units='m**3')
        
        # self.declare_partials('*','*')#,method='fd')
        
        #coded partials
        #finite difference part
        self.declare_partials(['F_I_1','F_I_BP'],['L','r_1'], method='fd')# This is monte carlo
        self.declare_partials('F',['L','r_1'], method='fd')
        self.declare_partials('B',['L','r_1','E'],method='fd')
        # exact part
        self.declare_partials(['F_1_BP','F_BP_1'],['L','r_1'], method='exact')
        self.declare_partials(['Volume'],['r_1','s_SiC','s_RPC','s_INS','L'],method='exact')
        self.declare_partials(['V_RPC'],['r_1','s_SiC','s_RPC','L'],method='exact')
        self.declare_partials(['m'],['r_1','s_SiC','s_RPC','Mass_Flow'],method='exact')
        self.declare_partials(['Ac_INS','r_g','r_n'],['r_1','s_SiC','s_RPC','s_INS'],method='exact')
        self.declare_partials(['Ac_RPC'],['r_1','s_SiC','s_RPC'],method='exact')
        self.declare_partials(['Ac_SiC'], ['r_1','s_SiC'],method='exact')
        self.declare_partials(['dL','z_n'], 'L')
        
        self.declare_partials('k_RPC',['k_Air','k_SiC'],method='exact')
        
        self.declare_partials('h_',['k_Air','Mass_Flow','r_1','s_SiC','s_RPC'],method='exact')
        self.declare_partials('h_loss_z', ['h_loss','k_INS','s_INS'],method='exact')
        
        # self.declare_partials('h_loss_cav',method='exact')# obsolete output!
        
        self.linear_solver = om.ScipyKrylov()
        
        # print('initialization.setup')
        # print(mass_Flow['val'])
        
    def compute(self, inputs, outputs):
        
        # in fucntion param.s
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        disc_r = self.disc_r
        Nr = self.Nr
        # Nz = self.Nz
        # NN = self.NN
        
        r_1 = inputs['r_1']
        s_SiC = inputs['s_SiC']
        s_RPC = inputs['s_RPC']
        s_INS = inputs['s_INS']
        L = inputs['L']
        outputs['Volume'] =  pi*(r_1+s_SiC+s_RPC+s_INS)**2*L
        
        h_loss = inputs['h_loss']
        
        k_INS = inputs['k_INS']
        k_SiC = inputs['k_SiC']
        k_Air = inputs['k_Air']
        
        porosity = self.porosity#inputs['porosity']
        
        p = self.p #inputs['p']
        
        Mass_Flow =abs(inputs['Mass_Flow'])
        # print(Mass_Flow)
        
        D_nom = self.D_nom #inputs['D_nom']
        
        # Tamb = inputs['Tamb']
        
        E = inputs['E']
        
        # Grid limits
        r_2=r_1+s_SiC;
        r_3=r_2+s_RPC;
        r_4=r_3+s_INS;
        outputs['dL'] = dL = L/disc_z;
        dr_SiC=(r_2-r_1)/disc_SiC;
        dr_RPC=(r_3-r_2)/disc_RPC;
        dr_INS=(r_4-r_3)/disc_INS;
        # bound1=self.bound1
        # bound2=self.bound2
        
        # vertical convection loss
        outputs['h_loss_z'] = 1/(1/h_loss+s_INS/k_INS);
        
        # Effective Heat Conduction Coefficient
        v=k_Air/k_SiC;
        self.f=0.3823;
        mu=self.f*(porosity*v+(1-porosity))+(1-self.f)*(porosity/v+1-porosity)**(-1);
        # outputs['k_RPC'] = k_SiC*mu; #the difference between rosseland and p1
        outputs['k_RPC'] = k_SiC*mu*0.75;      
        
        # Average Heat Transfer Coefficient in RPC
        d0=self.d0
        d1=self.d1
        d2=self.d2
        d3=self.d3
        self.rho_in=p*10**5/(287*298);
        u_IN=Mass_Flow/(self.rho_in*pi*(r_3**2-r_2**2));
        self.c1=0.00001595283450644;
        Re_IN=u_IN*D_nom/self.c1
        self.Pr=Pr=0.67
        Nu_IN=d0+d1*Re_IN**d2*Pr**d3;
        h_IN=Nu_IN*k_Air/D_nom;
        self.rho_out=p*10**5/(287*1473);
        u_OUT=Mass_Flow/(self.rho_out*pi*(r_3**2-r_2**2));
        self.c2=0.00020511715999999998;
        Re_OUT=u_OUT*D_nom/self.c2
        Nu_OUT=d0+d1*Re_OUT**d2*Pr**d3;
        h_OUT=Nu_OUT*k_Air/D_nom;
        outputs['h_'] = self.h_ = (h_IN+h_OUT)/2;
        
        # Calculation of the internal loss coefficient
        # T_ave=(1473+Tamb)/2; #Burayı Kontrol Etmek Lazım!
        # Pr=0.68;
        # dvis=1.458*10**-6*T_ave**1.5/(T_ave+110.4);
        # dens=10**6/(287*T_ave);  #Burayı Kontrol Et
        # kvis=dvis/dens;
        # Gr=9.81*1/T_ave*r_1**3*(1473-293)/kvis**2;
        # Ra=Gr*Pr;
        # T0t=r_1*Ra/L;
        # Nu_loss_cav=(10**(1.256*log10(T0t)-0.343))/(T0t);
        # h_loss_cav=Nu_loss_cav*k_Air/(r_1);
        # !!! #
        h_loss_cav=0; #h_loss_cav is set to zero!
        outputs['h_loss_cav'] = h_loss_cav
        
        #Calculation of z-coordinates
        z_n = np.empty((1,disc_z+2),dtype=float)
        z_n[0,0]=0;
        
        for i in range(1,disc_z+1):
            # z_n[0,i]=z_n[0,0]+dL/2+(i-1)*dL;
            z_n[0,i] = dL/2+(i-1)*dL;
        
        z_n[0,disc_z+1]=L;
        
        #Calculation of r-coordinates (n=nodal, g=grid)
        r_n = np.ones((Nr,1),dtype=float)
        r_g = np.ones((Nr-3,1),dtype=float)
        r_n[0,0]=r_1;
        
        for i in range(1,disc_SiC+1):
            r_n[i,0] = r_n[0]+dr_SiC/2+(i-1)*dr_SiC;
            r_g[i-1,0]=r_1+(i-1)*dr_SiC;
        
        r_n[disc_SiC+1,0]=r_2;
        r_g[disc_SiC,0]=r_2;
        
        for i in range(disc_SiC+2,(disc_SiC+2+disc_RPC)):
            r_n[i,0]=r_2+dr_RPC/2+(i-(disc_SiC+2))*dr_RPC;
            r_g[i-1,0]=r_2+(i-(disc_SiC+1))*dr_RPC;
        
        r_n[disc_SiC+2+disc_RPC,0]=r_3;
        
        for i in range (disc_SiC+2+disc_RPC+1,Nr-1):
            r_n[i,0]=r_3+dr_INS/2+(i-(disc_SiC+2+disc_RPC+1))*dr_INS;
            if i-1<=disc_r+1:
                r_g[i-2,0]=r_3+(i-(disc_SiC+disc_RPC+2))*dr_INS;
        
        r_n[Nr-1,0]=r_4;
        
        outputs['z_n'] = z_n
        outputs['r_n'] = r_n
        outputs['r_g'] = r_g
        
        # Calculation of cross-sectional areas
        Ac_SiC = np.empty((disc_SiC,1),dtype=float)
        for i in range (1,disc_SiC+1):
            Ac_SiC[i-1,0]=pi*(r_g[i]**2-r_g[i-1]**2);
        
        Ac_RPC = np.empty((disc_RPC,1),dtype=float)
        V_RPC = np.empty((disc_RPC,1),dtype=float)
        m = np.empty((disc_RPC,1),dtype=float)
        for i in range(1,disc_RPC+1):
            Ac_RPC[i-1,0]=pi*(r_g[disc_SiC+i]**2-r_g[disc_SiC+i-1]**2);
            V_RPC[i-1,0]=dL*Ac_RPC[i-1,0];
            m[i-1,0]=Mass_Flow*Ac_RPC[i-1,0]/(pi*(r_3**2-r_2**2));
            
        outputs['V_RPC'] = V_RPC
        outputs['m'] = self.m = m
            
        Ac_INS = np.empty((disc_INS,1),dtype=float)
        for i in range(1,disc_INS+1):
            Ac_INS[i-1,0]=pi*(r_g[disc_SiC+disc_RPC+i]**2-r_g[disc_SiC+disc_RPC+i-1]**2);
            
        outputs['Ac_SiC'] = Ac_SiC
        outputs['Ac_RPC'] =self.Ac_RPC= Ac_RPC
        outputs['Ac_INS'] = Ac_INS
        
        # Calculation of the Configuration Factors
        F_temp = np.empty((1,disc_z-1),dtype=float)
        
        for i in range(1,disc_z):
            L1=dL/r_1;
            L2=(dL+(i-1)*dL)/r_1;
            L3=(2*dL+(i-1)*dL)/r_1;
            X1=((L3-L1)**2+4)**0.5;
            X2=((L2-L1)**2+4)**0.5;
            X3=((L3)**2+4)**0.5;
            X4=((L2)**2+4)**0.5;
            F_temp[0,i-1]=1/(4*(L3-L2))*(2*L1*(L3-L2)+(L3-L1)*X1-(L2-L1)*X2-L3*X3+L2*X4);
        
        F_1_1=(1+dL/(2*r_1))-(1+(dL/(2*r_1))**2)**(0.5);
        F=np.zeros((disc_z,disc_z),dtype=float)
        np.fill_diagonal(F,F_1_1)
        
        for i in range(1,disc_z+1):
            w=i+1;
            e=1;
            neg=1;
            for j in range(1,disc_z):
                if w>disc_z:
                    w=1;
                    e=disc_z-j;
                    neg=-1;
                F[i-1,w-1]=F_temp[0,e-1];
                w=w+1;
                e=e+1*neg;
        
        del neg, e, w, F_1_1, F_temp
        
        F_1_I = np.empty((1,disc_z),dtype=float)
        # F_1_BP = np.empty((1,disc_z),dtype=float)
        for i in range(1,disc_z+1):
            j=disc_z-i+1;
            H1=dL/r_1;
            H2=(i-1)*dL/r_1;
            F_1_I[0,i-1]=0.25*((1+H2/H1)*(4+(H1+H2)**2)**0.5-(H1+2*H2)-H2/H1*(4+H2**2)**0.5);
            self.F_1_BP[0,j-1]=0.25*((1+H2/H1)*(4+(H1+H2)**2)**0.5-(H1+2*H2)-H2/H1*(4+H2**2)**0.5);
        
        F_BP_1=2*pi*r_1*dL/(pi*r_1**2)*np.copy(self.F_1_BP);
        F_1_1=F[0,0];
        
        # Radiosity Matrix Calculation
        B = np.ones((disc_z,disc_z),dtype=float)*-F*(1-E)/E;
        np.fill_diagonal(B,1/E-F_1_1*(1-E)/E)
        B = np.append(B,(-F_BP_1*(1-E)/E),axis=0)
        B = np.append(B,np.zeros((disc_z+1,1),dtype=float),axis=1)
        B[0:disc_z,-1] = (-self.F_1_BP*(1-E)/E)
        B[-1,-1] = 1/E
        
        outputs['B'] = B
        
        # print('Monte Carlo Calculation running...')
        F_I_1, F_I_BP = MoCa_3D(disc_z=disc_z,r_1=r_1,L=L,dL=dL)
        # print('Finished with Monte Carlo!')

        outputs['F'] = F
        outputs['F_1_BP'] = self.F_1_BP  
        outputs['F_BP_1'] = F_BP_1
        outputs['F_I_1'] = F_I_1
        outputs['F_I_BP'] = F_I_BP

        # print('initialization.compute')
        
    def compute_partials(self, inputs, J):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        disc_r = self.disc_r
        Nr = self.Nr
        
        r_1 = inputs['r_1']
        s_SiC = inputs['s_SiC']
        s_RPC = inputs['s_RPC']
        s_INS = inputs['s_INS']
        
        r = r_1+s_SiC+s_RPC+s_INS
        area = pi*r**2
        L = inputs['L']
        J['dL','L'] = 1/disc_z
        
        dvdr = 2*pi*r*L
        J['Volume','L'] = area
        J['Volume','r_1'] = dvdr
        J['Volume','s_SiC'] = dvdr
        J['Volume','s_RPC'] = dvdr
        J['Volume','s_INS'] = dvdr
        
        h_loss = inputs['h_loss']
        
        k_INS = inputs['k_INS']
        k_SiC = inputs['k_SiC']
        k_Air = inputs['k_Air']
        
        # vertical convection loss
        J['h_loss_z','h_loss'] = k_INS**2/(s_INS*h_loss+k_INS)**2
        J['h_loss_z','s_INS'] = -k_INS*h_loss**2/(s_INS*h_loss+k_INS)**2
        J['h_loss_z','k_INS'] = s_INS*h_loss**2/(s_INS*h_loss+k_INS)**2
        J['z_n','L'][1:-1,0] = np.arange(disc_z)/disc_z+0.5/disc_z
        J['z_n','L'][-1,0] = 1
        
        # Effective Heat Conduction Coefficient
        J['k_RPC','k_Air'] = self.porosity*(self.f-(k_SiC**2*(self.f-1))/(k_Air*(self.porosity-1)-self.porosity*k_SiC)**2)
        J['k_RPC','k_SiC'] = (self.porosity-1)*(((k_Air**2*(self.f-1))/(k_Air*(1-self.porosity)+self.porosity*k_SiC)**2)-self.f)
        J['k_RPC','k_Air'] = 0.75*J['k_RPC','k_Air']
        J['k_RPC','k_SiC'] = 0.75*J['k_RPC','k_SiC']
        
        J['h_','k_Air'] = self.h_/k_Air
        
        D_nom = self.D_nom
        Pr = self.Pr
        d0=self.d0
        d1=self.d1
        d2=self.d2
        d3=self.d3
        c1 = self.c1
        c2 = self.c2
        Mass_Flow =abs(inputs['Mass_Flow'])
        r_2=r_1+s_SiC;
        r_3=r_2+s_RPC;
        r_4=r_3+s_INS;
        
        Prd3d1k = Pr**d3*d1*d2*k_Air
        
        J['h_','Mass_Flow'] = Prd3d1k/D_nom/Mass_Flow*((D_nom*Mass_Flow)/(pi*self.rho_in*self.c1*s_RPC*(r_3+r_2)))**d2
        J['h_','Mass_Flow'] = (J['h_','Mass_Flow']+Prd3d1k/D_nom/Mass_Flow*((D_nom*Mass_Flow)/(pi*self.rho_out*self.c2*s_RPC*(r_3+r_2)))**d2)*0.5
        
        J['h_','r_1']=-2*Prd3d1k/D_nom/(r_3+r_2)*(D_nom*Mass_Flow/pi/self.rho_in/self.c1/s_RPC/(r_3+r_2))**d2
        J['h_','r_1']=(-2*Prd3d1k/D_nom/(r_3+r_2)*(D_nom*Mass_Flow/pi/self.rho_out/self.c2/s_RPC/(r_3+r_2))**d2+J['h_','r_1'])*0.5
        
        J['h_','s_SiC']=J['h_','r_1']
        
        J['h_','s_RPC'] =-Prd3d1k*Mass_Flow*2*r_3*(D_nom*Mass_Flow/self.rho_in/self.c1/pi/(r_3**2-r_2**2))**(d2-1)/self.rho_in/self.c1/pi/(r_3**2-r_2**2)**2
        J['h_','s_RPC']=(-Prd3d1k*Mass_Flow*2*r_3*(D_nom*Mass_Flow/self.rho_out/self.c2/pi/(r_3**2-r_2**2))**(d2-1)/self.rho_out/self.c2/pi/(r_3**2-r_2**2)**2+J['h_','s_RPC'])*0.5

        J['r_g','r_1'] = np.ones(disc_INS+disc_RPC+disc_SiC+1)
        J['r_g','s_SiC'] = np.ones(disc_INS+disc_RPC+disc_SiC+1)
        J['r_g','s_SiC'][:disc_SiC,0]=np.arange(disc_SiC)/disc_SiC
        J['r_g','s_RPC'][disc_SiC:disc_SiC+disc_RPC+1,0] = np.arange(disc_RPC+1)/disc_RPC
        J['r_g','s_RPC'][-disc_INS:,0] = np.ones(disc_INS)
        J['r_g','s_INS'][-(disc_INS+1):,0] = np.arange(disc_INS+1)/disc_INS
        
        J['r_n','r_1'] = np.ones(Nr)
        J['r_n','s_SiC'] = np.ones(Nr)
        J['r_n','s_SiC'][0,0] = 0
        J['r_n','s_SiC'][1:disc_SiC+1,0]=np.arange(disc_SiC)/disc_SiC+0.5/disc_SiC
        J['r_n','s_RPC'][disc_SiC+2:disc_SiC+disc_RPC+2,0] = np.arange(disc_RPC)/disc_RPC+0.5/disc_RPC
        J['r_n','s_RPC'][-(disc_INS+2):,0] = np.ones(disc_INS+2)
        J['r_n','s_INS'][-(disc_INS+1):,0] = np.arange(disc_INS+1)/disc_INS+0.5/disc_INS
        J['r_n','s_INS'][-1,0]=1
        
        J['Ac_SiC','r_1'] = 2*pi*s_SiC/disc_SiC
        J['Ac_SiC','s_SiC'] = 2*pi/disc_SiC*(r_1+s_SiC/disc_SiC*(2*np.arange(1,disc_SiC+1)-1))
        J['Ac_RPC','r_1'] = 2*pi*s_RPC/disc_RPC
        J['Ac_RPC','s_SiC'] = J['Ac_RPC','r_1']
        J['Ac_RPC','s_RPC'] = 2*pi/disc_RPC*((r_1+s_SiC)+s_RPC/disc_RPC*(2*np.arange(1,disc_RPC+1)-1))
        J['Ac_INS','r_1'] = 2*pi*s_INS/disc_INS
        J['Ac_INS','s_SiC'] = J['Ac_INS','r_1']
        J['Ac_INS','s_RPC'] = J['Ac_INS','r_1']
        J['Ac_INS','s_INS'] = 2*pi/disc_INS*((r_1+s_SiC+s_RPC)+s_INS/disc_INS*(2*np.arange(1,disc_INS+1)-1))
        
        J['m','Mass_Flow'] = self.m/Mass_Flow
        J['m','r_1'] = Mass_Flow/pi/s_RPC*(J['Ac_RPC','r_1']/(r_3+r_2)-2/(r_3+r_2)**2*self.Ac_RPC)
        J['m','s_SiC'] = J['m','r_1']
        J['m','s_RPC'] = Mass_Flow/pi*(J['Ac_RPC','s_RPC']/(2*r_2*s_RPC+s_RPC**2)-self.Ac_RPC*(2*r_3)/(2*r_2*s_RPC+s_RPC**2)**2)
        
        J['V_RPC','L'] = self.Ac_RPC/disc_z
        J['V_RPC','r_1'] = L/disc_z*J['Ac_RPC','r_1']
        J['V_RPC','s_SiC'] = J['V_RPC','r_1']
        J['V_RPC','s_RPC'] = L/disc_z*J['Ac_RPC','s_RPC']
        dL = L/disc_z
        H1=dL/r_1;
        for i in range(1,disc_z+1):
            J['F_1_BP','L'][disc_z-i,0]=0.25*(i**3*H1/sqrt((i*H1)**2+4)-(i-1)**3*H1/sqrt(((i-1)*H1)**2+4)-(2*i-1))/r_1/disc_z
        J['F_1_BP','r_1'] = J['F_1_BP','L']*-L/r_1
        a = 2*dL/r_1
        J['F_BP_1','L']=a/L*self.F_1_BP.T+a*J['F_1_BP','L']
        J['F_BP_1','r_1']=a/-r_1*self.F_1_BP.T+a*J['F_1_BP','r_1']

if __name__ == "__main__":
    
    tic = time.time()
    
    p = om.Problem()
    p.model.add_subsystem('init', initialization(disc_z = 20, disc_SiC = 10, disc_RPC=20, disc_INS = 10))
    p.setup()
    
    p.run_model()
    
    dL=p.get_val('init.dL')
    # B=p.get_val('init.B')
    # F=p.get_val('init.F')
    # F_I_1=p.get_val('init.F_I_1')
    # F_I_BP=p.get_val('init.F_I_BP')
    # F_1_BP=p.get_val('init.F_1_BP')
    # F_BP_1=p.get_val('init.F_BP_1')
    h_ = p.get_val('init.h_')
    V_RPC = p.get_val('init.V_RPC')
    # m = p.get_val('init.m')
    # np.save('m.npy',m)
    r_n = p.get_val('init.r_n')
    r_g = p.get_val('init.r_g')
    z_n = p.get_val('init.z_n')
    # h_loss_cav = p.get_val('init.h_loss_cav')
    h_loss_z = p.get_val('init.h_loss_z')
    k_RPC = p.get_val('init.k_RPC')
    Ac_SiC = p.get_val('init.Ac_SiC')
    Ac_RPC = p.get_val('init.Ac_RPC')
    Ac_INS = p.get_val('init.Ac_INS')
    
    
    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)
    data = p.check_partials(compact_print=True, show_only_incorrect=True, step_calc='rel_element')
    data = data['init']
    
    data_ = data['F_BP_1','r_1']['J_fd']
    data_calc = data['F_BP_1','r_1']['J_fwd']
    
    err = data_ - data_calc
    
    max_=np.amax(err)
    min_=np.amin(err)