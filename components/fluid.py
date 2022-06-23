#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:47:08 2021

@author: tufan
"""

import openmdao.api as om
import numpy as np

class fluid(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int, default=20,desc='Discretization in z-direction')
        self.options.declare('disc_RPC', types=int, default=20,desc='RPC-Discretization in r-direction')
        
        # print('fluid.initialize')

    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        self.add_input('h_',val=75.,desc='Heat transfer coefficient inside RPC',units='W/(m**2*K)')#4.2686
        self.add_input('A_spec',val=500,desc='Specific Surface of the RPC',units='m**-1')
        self.add_input('V_RPC', shape=(disc_RPC,1), desc='Volume of each RPC element', units='m**3')
        self.add_input('m', shape=(disc_RPC,1), desc='mass flow rate passing inside each RPC element', units='kg/s')
        self.add_input('cp',val=1005.,desc='Specific Heat Capacity',units='J/(kg*K)')
        
        self.add_input('T_RPC', shape=(disc_RPC,disc_z),val=1300., desc='Temperature distribution of the PRC', units='K')
        self.add_input('T_fluid_in', val=300.,desc='Inlet temperature of air',units='K')
        
        self.add_output('T_fluid',val=300.0, shape=(disc_RPC,disc_z), desc='Temperature of the air', units='K')
        
        self.declare_partials('*','*',method='exact')
        # self.declare_partials('*','*',method='fd')
        self.linear_solver = om.ScipyKrylov()

        self.Tf = np.zeros((self.options['disc_RPC'],self.options['disc_z']), dtype=float)
        
        # print('fluid.setup')
    
    def compute(self, inputs, outputs):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        h_ = inputs['h_']
        A_spec = inputs['A_spec']
        V_RPC = inputs['V_RPC']
        m = inputs['m']
        cp = inputs['cp']
        
        T_RPC = inputs['T_RPC']
        T_IN = np.ones((disc_RPC,1), dtype=float)*inputs['T_fluid_in']
        self.del_T = np.zeros((disc_RPC,disc_z),dtype = float) #T_RPC-T_inlet for element scale
        
        T_OUT  = np.zeros((disc_RPC,1), dtype=float)
        
        for i in range(disc_z):
            self.del_T[:,i] = T_IN[:,0]
            for j in range(disc_RPC):
                T_OUT[j,0]=(2*h_*A_spec*T_RPC[j,i]*V_RPC[j,0]-T_IN[j,0]*(h_*A_spec*V_RPC[j,0]-2*m[j,0]*cp))/(h_*A_spec*V_RPC[j,0]+2*m[j,0]*cp);
                self.Tf[j,i]=(T_IN[j,0]+T_OUT[j,0])/2;
            
            T_IN=np.copy(T_OUT);

        outputs['T_fluid'] = self.Tf
        
        # print('fluid.compute')
        
    def compute_partials(self, inputs, J):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        h_ = inputs['h_']
        A_spec = inputs['A_spec']
        V_RPC = inputs['V_RPC']
        m = inputs['m']
        cp = inputs['cp']
        T_RPC = inputs['T_RPC']
        
        T_IN = np.ones((disc_RPC,1), dtype=float)*inputs['T_fluid_in']
        
        p = h_ * A_spec * V_RPC;
        k = 2 * m * cp ;
        
        k_kp2 = k/(k+p)**2
        p_kp2 = p/(k+p)**2
        

        
        JTfTfi= np.zeros((disc_RPC,disc_z),dtype = float)# is p/(p+k)*dTi/dTi^ and initially Ti = Ti^
        dTfodTfi = np.ones((disc_RPC,1),dtype = float)
        
        for i in range(disc_z):
            for j in range(disc_RPC):
                JTfTfi[j,i] = (k[j,0])/(k[j,0]+p[j,0])*dTfodTfi[j,0]
            dTfodTfi[:,0] = 2*JTfTfi[:,i] - dTfodTfi[:,0]
        
        J['T_fluid','T_fluid_in'] = JTfTfi.flatten()
        
        dTfdTrpc = np.zeros(disc_RPC,dtype=float)
            
        dTfdTrpc[0] = p[0,0]/(p[0,0]+k[0,0])
        dTfdTrpc[1] = dTfdTrpc[0]*2*k[0,0]/(p[0,0]+k[0,0])
        a = (-p[0,0]+k[0,0])/(p[0,0]+k[0,0])
        for i in range(2,disc_z):
            dTfdTrpc[i] = dTfdTrpc[i-1] * a
        
        for i in range(disc_z):
            J['T_fluid', 'T_RPC'][i:disc_z,i] = dTfdTrpc[0:disc_z-i]
        for i in range(1,disc_RPC):
            J['T_fluid', 'T_RPC'][i*disc_z:i*disc_z+disc_z,i*disc_z:i*disc_z+disc_z] = J['T_fluid', 'T_RPC'][:disc_z,:disc_z]
            
        
        kp = k+p
        k_p = k-p
        k3p = 3*k+p
        
        dTfdp_Ts_comp = np.zeros((disc_RPC,disc_z),dtype=float)
        dTfdp_Ti_comp = np.zeros((disc_RPC,disc_z),dtype=float)
        dTfdp = np.zeros((disc_RPC,disc_z),dtype=float)
        
        dTfdp_Ts_comp[:,0] = k_kp2[:,0]
        dTfdp_Ti_comp[:,0] = -k_kp2[:,0]
        if disc_z >2:
            dTfdp_Ts_comp[:,1] = (k_kp2*2*k_p/kp)[:,0]
            dTfdp_Ti_comp[:,1] = (-k_kp2*(3*k-p)/kp)[:,0]
            
        if disc_z >3:
            for i in range(2,disc_z):
                dTfdp_Ts_comp[:,i] = (2*k_kp2/kp**i*(k*k-2*i*k*p+p*p)*k_p**(i-2))[:,0]
                dTfdp_Ti_comp[:,i] = (-k_kp2*k_p**(i-1)/kp**i*((2*i+1)*k-p))[:,0]
                
        for i in range(disc_z):
            for j in range(i+1):
                dTfdp[:,i] = dTfdp[:,i]+T_RPC[:,j]*dTfdp_Ts_comp[:,i-j]
            dTfdp[:,i] = dTfdp[:,i]+dTfdp_Ti_comp[:,i]*inputs['T_fluid_in']
        
        dpda = p/A_spec 
        dTfda = np.zeros((disc_z,disc_RPC),dtype=float)
        dpdh = p/h_
        dTfdh = np.zeros((disc_z,disc_RPC),dtype=float)
        dpdV = p/V_RPC
        dTfdV = np.zeros((disc_z,disc_RPC),dtype=float)
        
        for i in range(disc_z):
            dTfda[:,i] = dTfdp[:,i]*dpda[:,0]
            dTfdh[:,i] = dTfdp[:,i]*dpdh[:,0]
            dTfdV[:,i] = dTfdp[:,i]*dpdV[:,0]
        
        J['T_fluid','A_spec'] = dTfda.flatten()
        J['T_fluid','h_'] = dTfdh.flatten()
        for i in range(disc_RPC):          
            J['T_fluid','V_RPC'][i*disc_z:(i+1)*disc_z,i] = dTfdV[i,:].T
        
        
        dTfdk_Ts_comp = np.zeros((disc_RPC,disc_z),dtype=float)
        dTfdk_Ti_comp = np.zeros((disc_RPC,disc_z),dtype=float)
        dTfdk = np.zeros((disc_RPC,disc_z),dtype=float)
        
        dTfdk_Ts_comp[:,0] = -p_kp2[:,0]
        dTfdk_Ti_comp[:,0] = p_kp2[:,0]
        
        if disc_z >1:
            dTfdk_Ts_comp[:,1] = (-2*p_kp2*k_p/kp)[:,0]
            for i in range(1,disc_z):
                dTfdk_Ti_comp[:,i] = (p_kp2*k_p**(i-1)/kp**i*((2*i+1)*k-p))[:,0]
        
        if disc_z>2:
            for i in range(2,disc_z):
                dTfdk_Ts_comp[:,i] = (-2*p_kp2*k_p**(i-2)/kp**i*(k**2-2*i*k*p+p**2))[:,0]
        
        
        for i in range(disc_z):
            for j in range(i+1):
                dTfdk[:,i] = dTfdk[:,i]+T_RPC[:,j]*dTfdk_Ts_comp[:,i-j]
            dTfdk[:,i] = dTfdk[:,i]+dTfdk_Ti_comp[:,i]*inputs['T_fluid_in']
        
        dkdc = k/cp 
        dTfdc = np.zeros((disc_z,disc_RPC),dtype=float)
        dkdm = k/m
        dTfdm = np.zeros((disc_z,disc_RPC),dtype=float)
        
        for i in range(disc_z):
            dTfdc[:,i] = dTfdk[:,i]*dkdc[:,0]
            dTfdm[:,i] = dTfdk[:,i]*dkdm[:,0]
        
        J['T_fluid','cp'] = dTfdc.flatten()
        for i in range(disc_RPC):          
            J['T_fluid','m'][i*disc_z:(i+1)*disc_z,i] = dTfdm[i,:].T
        
        # print('fluid.compute_partials')

if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('fluid',fluid(disc_z = 20, disc_RPC = 20))
    p.setup()
    
    p.set_val('fluid.V_RPC', np.load('V_RPC.npy'))
    p.set_val('fluid.m', np.load('m.npy'))
    TRPC = np.load('Tfluid.npy')
    p.set_val('fluid.T_RPC',np.load('Tfluid.npy'))
    
    p.run_model()
    
    T_f = p.get_val('fluid.T_fluid')
    
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False,step_calc='rel_element')#,step=10**-6)#, rel_err_tol=5e-6,abs_err_tol=1e-5)
    data = data['fluid']
    # data_T = data['T_fluid','A_spec']['J_fd']
    data_V = data['T_fluid','m']['J_fd']
    # data_V = np.resize(data_V,(20,20))
    data_V_calc = data['T_fluid','m']['J_fwd']
    # data_V_calc = np.resize(data_V_calc,(20,20))
    
    err = data_V-data_V_calc
    
    max_=np.amax(err)
    min_=np.amin(err)

    # Q=p.get_val('radiocity.Q')
    # print(Q)
        
    