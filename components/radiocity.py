#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 24 13:28:10 2021

@author: tufan
This model for cavitiy radiocity eqn. solution for receiver model
"""

import openmdao.api as om
import numpy as np
from math import pi
# import time

class radiocity(om.ExplicitComponent):

    
    def initialize(self):
        
        self.options.declare('disc_z', types=int, default=20, desc='Discretization in z-direction')
        self.sigma = 5.67*10**(-8)
        
        # print('radiocity.initialize')
        
    def setup(self):
        
        disc_z = self.options['disc_z']
        self.Nz=disc_z+2;
        self.K=np.zeros((disc_z+1,1),dtype=float)
        
        self.add_input('Q_Solar_In',val=1000, desc='Total Solar Power Input',units='W')
        
        self.add_input('F_I_1', shape=(1,disc_z), desc='View factor aperture to cavity')
        self.add_input('F_I_BP', desc='View factor aperture to back plate')
        self.add_input('F_1_BP', shape=(1,disc_z), desc='View factor cavity to BP')
        self.add_input('F_BP_1', shape=(1,disc_z), desc='View factor BP to cavity')
        self.add_input('F', shape=(disc_z,disc_z), desc='View factor cavity to cavity')
        
        # self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('dL', desc='Axial mesh length', units='m')
        
        self.add_input('B', shape=(disc_z+1,disc_z+1), desc='Coeff. matrix of the radiocity')
        
        "CAV surface Temperature + BP"
        self.add_input('T_cav',val=293.0, shape=(1,self.Nz), desc='Temperature of the inner surf of CAV', units='K')
        self.add_input('T_BP',val=293.0, desc='Temperature of the BP of CAV', units='K')
        self.add_output('Q',shape=(disc_z+1,1), desc='Heat input to the surface of CAV', units='W')
        
        self.declare_partials(of='*', wrt='*', method='exact')
        self.linear_solver = om.ScipyKrylov()
        
        # print('radiocity.setup')
    
    def compute(self, inputs, outputs):
        
        # in fucntion param.s
        disc_z = self.options['disc_z']
        Nz = self.Nz
        
        Q_Solar_In = inputs['Q_Solar_In']
        
        F_I_1 = inputs['F_I_1']
        F_I_BP = inputs['F_I_BP']
        F_1_BP = inputs['F_1_BP']
        F_BP_1 = inputs['F_BP_1']
        F = inputs['F']
        
        sigma = self.sigma#inputs['sigma']
        
        r_1 = inputs['r_1']
        dL = inputs['dL']
        a_cav = 2*pi*r_1*dL
        a_BP = pi*r_1**2
        
        B = inputs['B']
        self.B_inv = np.linalg.inv(B)
        
        T = inputs['T_cav']
        # print(T)
        T_BP = inputs['T_BP']
        
        # K = np.zeros((disc_z+1,1),dtype=float)
        for i in range(1,disc_z+1):
            self.K[i-1,0] = -Q_Solar_In*F_I_1[0,i-1]/a_cav+sigma*T[0,i]**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F[i-1,:]))-sigma*T_BP**4*F_1_BP[0,i-1]                
            
        self.K[disc_z,0] = -Q_Solar_In*F_I_BP/a_BP+sigma*T_BP**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F_BP_1[0,:]))
        
        self.B_inv_K=np.matmul(self.B_inv,(-self.K))
        
        outputs['Q'][-1,0] = a_BP*self.B_inv_K[-1,0];
        outputs['Q'][0:-1,0] = (self.B_inv_K[0:-1,0]*a_cav).reshape(1,len(self.B_inv_K[0:-1,0]))
        
        # print('radiocity.compute')
        
    def compute_partials(self, inputs, J):
        
        disc_z = self.options['disc_z']
        Nz = self.Nz
        
        Q_Solar_In = inputs['Q_Solar_In']
        
        F_I_1 = inputs['F_I_1']
        F_I_BP = inputs['F_I_BP']
        F_1_BP = inputs['F_1_BP']
        F_BP_1 = inputs['F_BP_1']
        F = inputs['F']
        
        sigma = self.sigma#inputs['sigma']
        
        r_1 = inputs['r_1']
        dL = inputs['dL']
        a_cav = 2*pi*r_1*dL
        p_cav = 2*pi*r_1
        a_BP = pi*r_1**2
        
        # B = inputs['B']

        T = inputs['T_cav']
        T_BP = inputs['T_BP']
        
        for i in range(disc_z+1):
            for j in range(disc_z+1):
                J['Q','B'][i,i*(disc_z+1)+j] = -self.B_inv_K[j,0]
        
        for i in range(disc_z):
            J['Q','Q_Solar_In'][i,0] = F_I_1[0,i]/a_cav
            # J['Q','sigma'][i,0] = T[0,i+1]**4-sum(np.multiply(np.power(T[0,0:Nz-2],4),F[i,:]))-T_BP**4*F_1_BP[0,i]
            J['Q','T_BP'][i,0] = -4*sigma*T_BP**3*F_1_BP[0,i]
            # J['Q','F_I_BP'][i,0] = 0
            J['Q','dL'][i,0] = Q_Solar_In*F_I_1[0,i]/a_cav/dL
            J['Q','r_1'][i,0] = Q_Solar_In*F_I_1[0,i]/a_cav/r_1
            # J['Q','F_BP_1'][i,0] = 0
            J['Q','F_1_BP'][i,i] = -sigma*T_BP**4
            J['Q','F_I_1'][i,i] = -Q_Solar_In/a_cav
            for j in range(Nz-2):
                J['Q','T_cav'][i,j+1] = -4*sigma*T[0,j+1]**3*F[i,j]
            J['Q','T_cav'][i,i+1] = 4*sigma*T[0,i+1]**3+J['Q','T_cav'][i,i+1]
            for j in range(disc_z):
                J['Q','F'][i,i*disc_z+j] = -sigma*T[0,j+1]**4
            
        J['Q','Q_Solar_In'][disc_z,0] = F_I_BP/a_BP
        # J['Q','sigma'][disc_z,0] = T_BP**4-sum(np.multiply(np.power(T[0,1:Nz-1],4),F_BP_1[0,:]))
        J['Q','T_BP'][disc_z,0] = 4*sigma*T_BP**3
        J['Q','F_I_BP'][disc_z,0] = Q_Solar_In/a_BP
        # J['Q','dL'][disc_z,0] = 0
        J['Q','r_1'][disc_z,0] = 2*Q_Solar_In*F_I_BP/a_BP/r_1
        for i in range(Nz-2):
            J['Q','F_BP_1'][disc_z,i] = -sigma*np.power(T[0,i+1],4)
        # J['Q','F_1_BP'][disc_z,0] = 0
        # J['Q','F_I_1'][disc_z,0] = 0
        for i in range(disc_z):
            J['Q','T_cav'][disc_z,i+1] = -4*sigma*T[0,i+1]**3*F_BP_1[0,i]
        # J['Q','F'][disc_z,0] = 0
        
        J['Q','Q_Solar_In'] = np.matmul(self.B_inv,J['Q','Q_Solar_In'])
        # J['Q','sigma'] = -np.matmul(self.B_inv,J['Q','sigma'])
        J['Q','T_BP'] = -np.matmul(self.B_inv,J['Q','T_BP'])
        J['Q','F_I_BP'] = np.matmul(self.B_inv,J['Q','F_I_BP'])
        J['Q','dL'] = -np.matmul(self.B_inv,J['Q','dL'])
        J['Q','r_1'] = np.matmul(self.B_inv,J['Q','r_1'])
        J['Q','F_BP_1'] = -np.matmul(self.B_inv,J['Q','F_BP_1'])
        J['Q','F_1_BP'] = -np.matmul(self.B_inv,J['Q','F_1_BP'])
        J['Q','F_I_1'] = -np.matmul(self.B_inv,J['Q','F_I_1'])
        J['Q','T_cav'] = np.matmul(self.B_inv,J['Q','T_cav'])
        J['Q','F'] = np.matmul(self.B_inv,J['Q','F'])
        J['Q','B']=np.matmul(self.B_inv,J['Q','B'])

        J['Q','Q_Solar_In'][0:-1,0] = a_cav*J['Q','Q_Solar_In'][0:-1,0]
        # J['Q','sigma'][0:-1,0] = a_cav*J['Q','sigma'][0:-1,0]
        J['Q','T_BP'][0:-1,0] = a_cav*J['Q','T_BP'][0:-1,0]
        J['Q','F_I_BP'][0:-1,0] = a_cav*J['Q','F_I_BP'][0:-1,0]
        J['Q','dL'][0:-1,0] = a_cav*J['Q','dL'][0:-1,0]+p_cav*self.B_inv_K[0:-1,0]
        J['Q','r_1'][0:-1,0] = -a_cav*J['Q','r_1'][0:-1,0]+(2*pi*dL)*self.B_inv_K[0:-1,0]
        J['Q','F_BP_1'][0:-1,:] = a_cav*J['Q','F_BP_1'][0:-1,:]
        J['Q','F_1_BP'][0:-1,:] = a_cav*J['Q','F_1_BP'][0:-1,:]
        J['Q','F_I_1'][0:-1,:] = a_cav*J['Q','F_I_1'][0:-1,:]
        J['Q','T_cav'][0:-1,:] = -a_cav*J['Q','T_cav'][0:-1,:]
        J['Q','F'][0:-1,:] = -a_cav*J['Q','F'][0:-1,:]
        J['Q','B'][0:-1,:] = a_cav*J['Q','B'][0:-1,:]
        
        J['Q','Q_Solar_In'][-1,0] = a_BP*J['Q','Q_Solar_In'][-1,0]
        # J['Q','sigma'][-1,0] = a_BP*J['Q','sigma'][-1,0]
        J['Q','T_BP'][-1,0] = a_BP*J['Q','T_BP'][-1,0]
        J['Q','F_I_BP'][-1,0] = a_BP*J['Q','F_I_BP'][-1,0]
        J['Q','dL'][-1,0] = a_BP*J['Q','dL'][-1,0]
        J['Q','r_1'][-1,0] = -a_BP*J['Q','r_1'][-1,0]+p_cav*self.B_inv_K[-1,0]
        J['Q','F_BP_1'][-1,:] = a_BP*J['Q','F_BP_1'][-1,:]
        J['Q','F_1_BP'][-1,:] = a_BP*J['Q','F_1_BP'][-1,:]
        J['Q','F_I_1'][-1,:] = a_BP*J['Q','F_I_1'][-1,:]
        J['Q','T_cav'][-1,:] = -a_BP*J['Q','T_cav'][-1,:]
        J['Q','F'][-1,:] = -a_BP*J['Q','F'][-1,:]
        J['Q','B'][-1,:] = a_BP*J['Q','B'][-1,:]
        
        # print("radiocity.partials")
        
if __name__ =='__main__':
    
    p = om.Problem()
    p.model.add_subsystem('radiocity', radiocity(disc_z = 20))
    p.setup()
    
    p.set_val('radiocity.F_I_1', np.load('F_I_1.npy'))
    p.set_val('radiocity.F_I_BP', np.load('F_I_BP.npy'))
    p.set_val('radiocity.F_1_BP', np.load('F_1_BP.npy'))
    p.set_val('radiocity.F_BP_1', np.load('F_BP_1.npy'))
    p.set_val('radiocity.F', np.load('F.npy'))
    p.set_val('radiocity.B', np.load('B.npy'))
    p.set_val('radiocity.dL', np.load('dL.npy'))
    p.set_val('radiocity.T_cav',np.load('T_cav.npy'))
    
    p.run_model()
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False)#, rel_err_tol=1e-4,abs_err_tol=0.5)
    data = data['radiocity']
    data_F = data['Q','F']['J_fd']
    data_B = data['Q','F_BP_1']['J_fd']
    data_B_calc = data['Q','F_BP_1']['J_fwd']
    
    err = data_B-data_B_calc
    
    max_=np.amax(err)
    min_=np.amin(err)

    Q=p.get_val('radiocity.Q')
    # print(Q)