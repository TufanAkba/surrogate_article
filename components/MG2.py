#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 03:22:18 2022

@author: tufanakba
new motor-generator configuration for electrical power and ang. vel. are 
inputs and torque is the output

For calculation every input positive but for sign convention, super should have:

For motoring, M>0 ==> P_el>0
For generation, M<0 ==> P_el<0

For motoring,
    P_el = P_mech + P_joule

For generation,
    P_el = P_mech - P_joule
    
P_el ==> input
P_mech = n/kn * I
P_joule = I**2 * R

M = km * I

Inputs: P_el, n ,kn , R, km
then calculate I,
it is a quadratic function and due to physical meaning different roots are used.

I used maxon motor 305015 for input data - no reason.
"""

import openmdao.api as om
from math import pi, sqrt

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
        # self.add_input('I0', val=356, units='A/1000', desc='No Load Current')
        # self.add_input('V', val=48, units='V', desc='Nominal Voltage')
        
        # inputs from the components or model
        self.add_input('n', val=15000, units='rpm', desc='Angular Speed')
        self.add_input('P_el',units='W', desc='Electrical Power')
        
        self.add_output('trq', units='N*mm', desc='Torque')
        self.add_output('P_shaft', units='W', desc='Shaft Power')
        
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
        n = inputs['n']
        P_el = inputs['P_el']
        #discriminant check:
        check = (n/kn)**2+4*R*P_el
        
        if check>0:
            
            outputs['trq'] = -km * (n/kn-sqrt(check))/(2*R)
            # self.trq = outputs['trq']
            # outputs['P_shaft'] = outputs['trq']/1000*n*pi/30
        else:
            outputs['trq'] = 0
            print('no generation in generation mode')
        
        # if P_el < 0:
            
        #     # P_el = -P_el
        #     # I = (n/kn-sqrt((n/kn)**2-4*R*P_el))/(2*R)
            
            
        #     if check>0:
        #         outputs['M'] = -km * (n/kn-sqrt(check))/(2*R)
        #     else:
        #         outputs['M'] = 0
        #         print('no generation in generation mode')
            
        # else:
            
        #     # I = (-n/kn+sqrt((n/kn)**2+4*R*P_el))/(2*R)
        #     outputs['M'] = km * (-n/kn+sqrt(check))/(2*R)
            
        # outputs['M'] = km * (-n/kn+sqrt(check))/(2*R)
        
        # print('MG.compute')
        
    def compute_partials(self, inputs, J):
        
        km = inputs['km']
        kn = inputs['kn']
        R = inputs['R']
        n = inputs['n']
        P_el = inputs['P_el']
        nkn = n/kn
        check = (nkn)**2+4*R*P_el
        
        if check>0:
            disc = sqrt((n/kn)**2+4*R*P_el)

            J['trq','km'] = -(nkn-(disc))/(2*R)
            J['trq','kn'] = km*n/2/kn**2/R*(1-nkn/(disc))
            J['trq','n'] = km/2/kn/R*(nkn/(disc)-1)
            J['trq','R'] = km*P_el/R/(disc)-km*((disc)-nkn)/2/R**2
            J['trq','P_el'] = km/(disc)
            
            # const = pi/30./1000.*n
            
            # J['P_shaft','km'] = J['trq','km']*const
            # J['P_shaft','kn'] = J['trq','kn']*const
            # J['P_shaft','n'] = (J['trq','n']*n + self.trq)*pi/30/1000
            # J['P_shaft','R'] = J['trq','R']*const
            # J['P_shaft','P_el'] = J['trq','P_el']*const
        
        else:
            J = 0
        
    # print('MG.partials')
if __name__ == '__main__':  
    p = om.Problem()
    p.model.add_subsystem('MG', MG())
    p.setup()
    p.set_val('MG.P_el',-1) # in default motor vals, -10 fails
    
    p.run_model()
    
    print(p.get_val('MG.trq'))
    print(p.get_val('MG.P_shaft'))
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=1e-4,abs_err_tol=1e-4)
    