#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:54:07 2021

@author: tufanakba
explicit component for T_BP
"""

import openmdao.api as om
import numpy as np
from math import pi

class T_BP(om.ExplicitComponent):
    
    def initialize(self):
        
        self.sigma = 5.67*10**(-8)
        
        # print('T_BP.initialize')
    
    def setup(self):
        
        self.add_input('Q', desc='Heat input to the BP of CAV', units='W')
        self.add_input('E',val=0.9,desc='Emissivity')
        # self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        
        self.add_output('T_BP', units='K', desc='BP temperature')
        
        self.declare_partials(of='*', wrt='*',method='exact')
        # self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('T_BP.setup')

    def compute(self, inputs, outputs):

        E = inputs['E']
        if inputs['Q']<0:
            Q_Solar_Net = inputs['Q']*-1
            print('neg Q')
        else:
            Q_Solar_Net = inputs['Q']
        r_1 = inputs['r_1']
        
        outputs['T_BP'] = self.T_BP = (Q_Solar_Net/(pi*r_1**2*self.sigma*E))**0.25
        # print(self.T_BP)
        
        # print('T_BP.compute')
        
    def compute_partials(self, inputs, J):
        
        E = inputs['E']
        Q_Solar_Net = abs(inputs['Q'])
        r_1 = inputs['r_1']
        
        J['T_BP','Q'] = self.T_BP*0.25/Q_Solar_Net #0.25*(Q_Solar_Net_new_1[-1,0]**-0.75)/((pi*r_1**2*sigma*E))**0.25
        J['T_BP','E'] = self.T_BP * -0.25/E #-0.25/(E**1.25)*(Q_Solar_Net_new_1[-1,0]/(pi*r_1**2*sigma))**0.25
        # J['T_BP','sigma'] = T_BP* -0.25/sigma #-0.25/(sigma**1.25)*(Q_Solar_Net_new_1[-1,0]/(pi*r_1**2*E))**0.25
        J['T_BP','r_1'] = self.T_BP * -0.5 /r_1 #-0.5/(r_1**1.5)*(Q_Solar_Net_new_1[-1,0]/(pi*sigma*E))**0.25
        
       
    
if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('T_BP', T_BP())
    p.setup(force_alloc_complex=True)
    
    p.set_val('T_BP.Q', 200)

    
    p.run_model()
    data = p.check_partials(compact_print=False)
    # data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=5e-3,abs_err_tol=0.6)
    data = data['T_BP']
    data_A = data['T_BP','Q']['J_fd']
    data_A_calc = data['T_BP','Q']['J_fwd']
    
    err = data_A-data_A_calc
    
    max_=np.amax(err)
    min_=np.amin(err)

    T_BP=p.get_val('T_BP.T_BP')
    print(T_BP)