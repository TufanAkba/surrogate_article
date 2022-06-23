#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:47:06 2021

@author: tufanakba

This component calculates outlet fluid temperature
"""

import openmdao.api as om
import numpy as np

class Tf_out(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_RPC', types=int, default=20,desc='RPC-Discretization in r-direction')
        
        # print('Tf_out.initialize')

    def setup(self):
        
        disc_RPC = self.options['disc_RPC']
        
        self.add_input('m', shape=(disc_RPC,1), desc='mass flow rate passing inside each RPC element', units='kg/s')
        self.add_input('T_fluid',val=300.0, shape=(disc_RPC,1), desc='Temperature of the air', units='K')
        
        self.add_output('T_fluid_out', val=293.0, desc='Outlet temperature of air',units='K')
        
        # for efficiency calculations
        self.add_input('cp',val=1005.,desc='Specific Heat Capacity',units='J/(kg*K)')
        self.add_input('T_fluid_in', val=300.,desc='Inlet temperature of air',units='K')
        self.add_input('Q_Solar_In',val=1000., desc='Total Solar Power Input',units='W')
        self.add_output('eff_S2G', desc='Efficiency solar input to outlet')
        
        self.declare_partials('eff_S2G', '*',method='exact')
        self.declare_partials('T_fluid_out',['T_fluid','m'], method='exact')
        # self.declare_partials('eff_S2G', '*',method='fd')
        # self.declare_partials('T_fluid_out',['T_fluid','m'], method='fd')

        self.linear_solver = om.ScipyKrylov()
        
        # print('Tf_out.setup')
    
    def compute(self, inputs, outputs):

        m = inputs['m']
        T_fluid = inputs['T_fluid']

        self.T_out = sum(np.multiply(m,T_fluid)/np.sum(m));
        outputs['T_fluid_out'] = self.T_out
        # print(self.T_out)
        
        cp = inputs['cp']
        self.Q_Fluid=cp*sum(np.multiply(m, (self.T_out-inputs['T_fluid_in'])))

        self.eff = (self.Q_Fluid/inputs['Q_Solar_In'])
        outputs['eff_S2G'] = self.eff
        
        # print(f'{self.T_out}\t{self.eff}')
        
        # print('Tf_out.compute')
        
    def compute_partials(self, inputs, J):
        
        m = inputs['m']
        T_fluid = inputs['T_fluid']
        m_sum = sum(m)
        
        J['T_fluid_out','T_fluid'] = m/m_sum
        J['T_fluid_out','m'][0,:] = (T_fluid[:,0]-self.T_out)/m_sum
        
        cp = inputs['cp']
        Q_Solar_In = inputs['Q_Solar_In']
        
        J['eff_S2G', 'Q_Solar_In'] = -self.Q_Fluid/Q_Solar_In**2
        J['eff_S2G', 'T_fluid_in'] = -cp*m_sum/Q_Solar_In
        J['eff_S2G', 'cp'] = self.eff/cp
        J['eff_S2G', 'T_fluid'][0,:] = cp*m[:,0]/Q_Solar_In
        
        T_fluid = T_fluid-inputs['T_fluid_in']
        J['eff_S2G', 'm'][0,:] = cp/Q_Solar_In*T_fluid[:,0]

if __name__ == '__main__':
    
    p = om.Problem()
    disc_RPC = 20
    p.model.add_subsystem('Tf_out',Tf_out(disc_RPC = disc_RPC))
    p.setup()
    
    
    m = np.load('m.npy')
    p.set_val('Tf_out.m', m)
    
    T_fluid = np.arange(disc_RPC,dtype=float)+300
    p.set_val('Tf_out.T_fluid',T_fluid)
    
    p.run_model()
    
    T_f = p.get_val('Tf_out.T_fluid_out')
    
    
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=5e-6,abs_err_tol=1e-5,step_calc='rel_element')
    data = data['Tf_out']

    # data_m = data['T_fluid_out','m']['J_fd']
    # data_m_calc = data['T_fluid_out','m']['J_fwd']

    # data_T = data['T_fluid_out','T_fluid']['J_fd']
    # data_T_calc = data['T_fluid_out','T_fluid']['J_fwd']
    
    data_eff = data['eff_S2G','m']['J_fd']
    data_eff_calc = data['eff_S2G','m']['J_fwd']
    
    err = data_eff-data_eff_calc
    
    max_=np.amax(err)
    min_=np.amin(err)