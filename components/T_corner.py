#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:19:10 2021

@author: tufan
"""

import openmdao.api as om
import numpy as np

class T_corner(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.bound1 = self.options['disc_SiC']+2;
        
        # print('T_corner.initialize')
    
    def setup(self):
        
        self.add_input('T_side',val=293.0, shape=(self.options['disc_SiC'],1), desc='Temperature of the verical surf of SiC', units='K')
        self.add_input('Ac_SiC', shape=(self.options['disc_SiC'],1), units='m**2', desc='Cross sectional area of SiC elements')
        
        self.add_output('T_corner', units='K', desc='Corner temperature')
        
        self.declare_partials(of='*', wrt='*',method='exact')
        # self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('T_corner.setup')

    def compute(self, inputs, outputs):

        # outputs['T_corner']=sum((np.multiply(inputs['Ac_SiC'],inputs['T_side'][:,0])).diagonal())/sum(inputs['Ac_SiC'])
        outputs['T_corner']=sum((np.multiply(inputs['Ac_SiC'],inputs['T_side'])))/sum(inputs['Ac_SiC'])
        
        # print('T_corner.compute')
        
    def compute_partials(self, inputs, J):
        
        T = inputs['T_side']
        A = inputs['Ac_SiC']
        A_sum = sum(inputs['Ac_SiC'])
        AT_sum = sum((np.multiply(inputs['Ac_SiC'],inputs['T_side'])))
        
        for i in range(self.options['disc_SiC']):
            J['T_corner','T_side'][0,i] = A[i]/A_sum
            J['T_corner','Ac_SiC'][0,i] = -(AT_sum - T[i]*A_sum)/A_sum**2
       
    
if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('T_corner', T_corner(disc_SiC = 10))
    p.setup(force_alloc_complex=True)
    
    p.set_val('T_corner.Ac_SiC', np.load('Ac_SiC.npy'))
    # p.set_val('radiocity.F_I_BP', np.load('F_I_BP.npy'))
    # p.set_val('radiocity.F_1_BP', np.load('F_1_BP.npy'))
    # p.set_val('radiocity.F_BP_1', np.load('F_BP_1.npy'))
    # p.set_val('radiocity.F', np.load('F.npy'))
    # p.set_val('radiocity.B', np.load('B.npy'))
    # p.set_val('radiocity.dL', np.load('dL.npy'))
    
    p.run_model()
    # data = p.check_partials(compact_print=False)
    data = p.check_partials(compact_print=True, show_only_incorrect=False, rel_err_tol=5e-3,abs_err_tol=0.6)
    data = data['T_corner']
    data_T = data['T_corner','T_side']['J_fd']
    data_A = data['T_corner','Ac_SiC']['J_fd']
    data_A_calc = data['T_corner','Ac_SiC']['J_fwd']
    
    err = data_A-data_A_calc
    
    max_=np.amax(err)
    min_=np.amin(err)