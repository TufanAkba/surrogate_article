#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:30:47 2022

@author: tufanakba

Motor-Generator is an explicit component for calculating the electrical power
input or output

Motor Parameters:
    kn = speed constant (rpm/V)
    km = torque constant (N*mm/A)
    note that kn*km = 300000/pi

For motoring: It is actully implicit in super class
    desired shaft torque and rpm calculate electrical power
    P_mech = pi/30000*M*n
    P_el = P_mech + P_joule
    
    where,
    P_joule = R * I**2 and I = M/km
    
    P_el = pi/30000*M*n + R*I**2
    
For generator: It is explicit
    given shaft and torque generated electrical power is calculated
    from given moment
    M = km * (IL + I0) --> IL = M/km-I0
    
    P_el = IL * V_applied = IL * (n/kn - R*IL)

"""

import openmdao.api as om
from math import pi

class MG(om.ExplicitComponent):
    
    # def initialize(self):
        
        # self.options.declare('Print',types=bool,default=True)
        
        # self.options.declare('Generator', types=bool, default=True,
                             # desc='True generator mode, False motor mode')
                             
                        
        
        # print('MG.initialize')
    
    def setup(self):
        
        # default values for Maxon 305015 Brushless DC Motor
        self.add_input('km', val=27.6, units='N*mm/A', desc='Torque Constant')
        self.add_input('kn', val=346, units='rpm/V', desc='Speed Constant')
        self.add_input('R', val=0.386, units='ohm', desc='Terminal Resistance')
        self.add_input('I0', val=356, units='A/1000', desc='No Load Current')
        # self.add_input('V', val=48, units='V',
        #                desc='Nominal Voltage')
        
        # moment and speed values from the shaft
        self.add_input('n', val=1000, units='rpm', desc='Angular Speed')
        self.add_input('M', val=100, units='N*mm', desc='Torque')
        
        self.add_output('P_el',units='W', desc='Electrical Power')
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
        kn = inputs['kn']
        R = inputs['R']
        I0 = inputs['I0']
        # V = inputs['V']
        
        n = inputs['n']
        M = inputs['M']
        
        # if self.options['Generator']:
        if M>0:
            # if self.options['Print']:
            #     self.options['Print'] = False
            #     print(f'{M} in generation')
            IL = M/km - I0
            outputs['P_el'] = IL * (n/kn - R*IL)
            
        else:
            # if self.options['Print']:
            #     self.options['Print'] = False
            #     print(f'{M} in motoring')
            outputs['P_el'] = pi/30000*M*n - R* (M/km)**2
            
    def compute_partials(self, inputs, J):
        
        km = inputs['km']
        kn = inputs['kn']
        R = inputs['R']
        I0 = inputs['I0']
        # V = inputs['V']
        
        n = inputs['n']
        M = inputs['M']
        
        if M>0:
            IL = M/km - I0
            dpdIL=n/kn-2*R*IL
            
            J['P_el','M'] = dpdIL/km
            J['P_el','km'] = dpdIL*-M/km**2
            J['P_el','I0'] = -dpdIL
            J['P_el','n'] = IL/kn
            J['P_el','kn'] = -IL*n/kn**2
            J['P_el','R'] = -IL**2
        
        else:
            
            J['P_el','M'] = pi/30000*n-2*R*M/km**2
            J['P_el','km'] = 2*R*M**2/km**3
            # J['P_el','I0'] = 
            J['P_el','n'] = pi/30000*M
            # J['P_el','kn'] = 
            J['P_el','R'] = -(M/km)**2
            
if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('MG', MG())
    p.setup()
    p.set_val('MG.M',100)
    
    p.run_model()
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=1e-4,abs_err_tol=1e-4)
    
    p2 = om.Problem()
    p2.model.add_subsystem('MG', MG())
    p2.setup()
    p2.set_val('MG.M',-100)
    
    p2.run_model()
    # data = p.check_partials(compact_print=False)
    data = p2.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=1e-4,abs_err_tol=1e-4)
    
