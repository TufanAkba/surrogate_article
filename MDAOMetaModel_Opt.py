#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:18:52 2022

@author: tufanakba
Meta model test for optimization
"""

import numpy as np
import openmdao.api as om

class surr(om.MetaModelUnStructuredComp):
    
    def initialize(self):
        self.options.declare('test_folder',default='test_folder',
                             desc='Test folder location of resp. surf.')
        self.options.declare('kriging',default=False,
                             desc='Select kriging or response surface method')
        super().initialize()
    
    def setup(self):
        
        test_folder = self.options['test_folder']

        self.add_input('Tfi',training_data=np.loadtxt(test_folder + '/Tfi.csv'), units='K')
        self.add_input('m_dot',training_data=np.loadtxt(test_folder + '/m_dot.csv'), units='kg/s')
        self.add_input('rpc',training_data=np.loadtxt(test_folder + '/rpc.csv'), units='m')
        self.add_input('ins',training_data=np.loadtxt(test_folder + '/ins.csv'), units='m')
        self.add_input('L',training_data=np.loadtxt(test_folder + '/L.csv'), units='m')
        
        if self.options['kriging']:
            self.add_output('vol',training_data=np.loadtxt(test_folder + '/vol.csv'), surrogate=om.KrigingSurrogate(eval_rmse= True), units='m**3')
            self.add_output('Tfo',training_data=np.loadtxt(test_folder + '/Tfo.csv'), surrogate=om.KrigingSurrogate(eval_rmse= True), units='K')
            self.add_output('T_o',training_data=np.loadtxt(test_folder + '/T_o.csv'), surrogate=om.KrigingSurrogate(eval_rmse= True), units='K')
        
        else:
            self.add_output('vol',training_data=np.loadtxt(test_folder + '/vol.csv'), surrogate=om.ResponseSurface(), units='m**3')
            self.add_output('Tfo',training_data=np.loadtxt(test_folder + '/Tfo.csv'), surrogate=om.ResponseSurface(), units='K')
            self.add_output('T_o',training_data=np.loadtxt(test_folder + '/T_o.csv'), surrogate=om.ResponseSurface(), units='K')
            
        self.declare_partials(['Tfo','T_o'],'*', method='fd')
        self.declare_partials('vol', ['rpc','ins','L'], method='fd')
        
        super().setup()
        
class surrOpt(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('L', types=float, default=0.065, desc='Lenght of receiver for optimization')
        self.options.declare('s_RPC', types=float, default=0.015, desc='RPC thickness of receiver for optimization')
        self.options.declare('s_INS', types=float, default=0.1, desc='INS thickness of receiver for optimization')
        self.options.declare('test_folder',default='test_folder', desc='Test folder location of resp. surf.')
        self.options.declare('kriging',default=False, desc='Selecting kriging or response surface for surrogate')
    
    def setup(self):     
        
        #optimizer params
        optimizer='SLSQP'
        opt_tol = 1e-8 #do not set less than 1e-5 def -6
        opt_maxiter = 100
        debug_print=[]
        
        self.add_input('Tfi',300., desc='Inlet temperature of air', units='K')
        self.add_input('m_dot',0.0006, desc='Mass flow rate', units='kg/s')
        self.add_output('rpc', desc='Thikness of RPC tube', units='m')
        self.add_output('ins', desc='Thikness of INS tube', units='m')
        self.add_output('L', desc='Thikness of RPC tube', units='m')
        
        self.add_output('Tfo', desc='Outlet temperature of air',units='K')
        
        # debug_print=[]
        debug_print = ['desvars','objs','nl_cons','totals']
        
        self._problem = prob =  om.Problem()
        prob.model.add_subsystem('surr', surr(test_folder=self.options['test_folder'],
                                              kriging=self.options['kriging'])
                                 ,promotes=['*'])
        prob.driver = om.ScipyOptimizeDriver(debug_print = debug_print,
                                             optimizer=optimizer, 
                                             tol=opt_tol,
                                             maxiter = opt_maxiter)
        
        prob.model.set_input_defaults('rpc', self.options['s_RPC'], units='m')
        prob.model.set_input_defaults('ins', self.options['s_INS'], units='m')
        prob.model.set_input_defaults('L', self.options['L'], units='m')
        
        prob.model.add_design_var('rpc',lower=0.005, upper=0.025)
        prob.model.add_design_var('ins',lower=0.05, upper=0.25)
        prob.model.add_design_var('L',lower=0.02, upper=0.1)
        
        prob.model.add_constraint('T_o', upper=373, lower=353, scaler=0.02)
        prob.model.add_constraint('vol',lower=0.002, upper=0.003721609197258809, scaler=100)
        
        prob.model.add_objective('Tfo',scaler=-0.01)
        self.declare_partials('*', '*',method='fd',step_calc='rel_avg')
        prob.setup()
        
    def compute(self, inputs, outputs):
        
        prob = self._problem
        prob.set_val('m_dot', inputs['m_dot'], units='kg/s')
        prob.set_val('Tfi',inputs['Tfi'], units='K')
        
        
        prob.final_setup()
        prob.run_driver()
        
        outputs['rpc'] = prob['rpc']
        outputs['ins'] = prob['ins']
        outputs['L'] = prob['L']
        outputs['Tfo'] = prob['Tfo']
        
        if kriging:
            print('rmse = ', prob.model.surr._metadata('Tfo')['rmse'][0, 0])
            

if __name__ == "__main__":
    
    import time
    
    st = time.time()
    
    p = om.Problem()
    
    kriging = False
    
    p.model.add_subsystem('receiver', surrOpt(test_folder='4_5',kriging=kriging))
    
    p.setup()
    
    p.set_val('receiver.m_dot',0.00066, units='kg/s') #rearrange to 66
    
    p.run_model()
    
    p.model.list_outputs(units=True,prom_name=True,shape=False)
    p.model.list_inputs(units=True,prom_name=True,shape=False)

    print("time", time.time() - st)
