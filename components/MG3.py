#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:40:10 2022

@author: tufanakba
Motor-generator version 3
Shaft power, torque, n are inputs
Electrical power is output

Sign convention is important!..
For generation, torque input is positive thus Pow_el positive
For motoring, torque is negative thus Pow_el negative

Maxon 305015 model is implemented for analysis
"""

import openmdao.api as om
from math import pi, sqrt

class MG(om.ExplicitComponent):
    
    def initialize(self):
        
        # self.options.declare('Print',types=bool,default=True)
        
        # self.options.declare('Generator', types=bool, default=True,
        #                       desc='True generator mode, False motor mode')
                             
                        
        
        # print('MG.initialize')
        
        self.kmkn = 30/pi
    
    def setup(self):
        
        # default values for Maxon 305015 Brushless DC Motor
        self.add_input('km', val=0.0276, units='N*m/A', desc='Torque Constant')
        # self.add_input('kn', val=346, units='rpm/V', desc='Speed Constant')
        self.add_input('R', val=0.386, units='ohm', desc='Terminal Resistance')
        # self.add_input('I0', val=356, units='A/1000', desc='No Load Current')
        # self.add_input('V', val=48, units='V', desc='Nominal Voltage')
        
        # inputs from the components or model
        self.add_input('n', val=15000, units='rpm', desc='Angular Speed')
        self.add_output('P_el',units='W', desc='Electrical Power')
        
        self.add_input('trq', units='N*m', desc='Torque')
        # self.add_input('P_shaft', units='W', desc='Shaft Power')
        
        self.declare_partials(of='*', wrt='*',method='exact')
        # if self.options['Generator']:
            # self.declare_partials(of='*', wrt='*',method='exact')
        # else:
            # self.declare_partials(of='*', wrt=['M','R','km','n'],method='exact')
        # self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('MG.setup')
        
    def compute(self, inputs, outputs):
        
        km = inputs['km']
        # kn = inputs['kn']
        R = inputs['R']
        n = inputs['n']
        
        M = inputs['trq']
        
        outputs['P_el'] = n/self.kmkn*M-R*(M/km)**2
        
    def compute_partials(self, inputs, J):
        
        km = inputs['km']
        # kn = inputs['kn']
        R = inputs['R']
        n = inputs['n']
        
        M = inputs['trq']
        
        J['P_el','n'] = M/self.kmkn
        J['P_el','trq'] = n/self.kmkn - 2*R*M/km**2
        J['P_el','R'] = -(M/km)**2
        J['P_el','km'] = 2*R*M**2/km**3
        
        
if __name__ == '__main__':
    
    
    p = om.Problem()
    p.model.add_subsystem('MG', MG())
    p.setup()
    
    p.set_val('MG.trq',0.09872)
    
    p.run_model()
    
    print(p.get_val('MG.P_el', units='W'))
    
    data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=1e-4,abs_err_tol=1e-4, step_calc='rel_element')