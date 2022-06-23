#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:09:51 2021

@author: tufan
p1 analytic derivative model
"""

import openmdao.api as om
import numpy as np
# import time

from components.initialization import initialization
from components.I_Rad import I_Rad
from components.radiocity import radiocity
from components.solid import solid
from components.T_corner import T_corner
from components.fluid import fluid
from components.Tf_out import Tf_out
from components.T_BP import T_BP

from components.draw_contour import draw_contour

class Receiver(om.Group):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        # print('Receiver.initialize')

        
    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        subsys = initialization(disc_z = disc_z, disc_SiC = disc_SiC, disc_INS = disc_INS, disc_RPC = disc_RPC)
        self.add_subsystem('init', subsys, 
                           promotes_inputs=['*'], 
                           promotes_outputs=['*'])
        
        subsys = radiocity(disc_z = disc_z)
        self.add_subsystem('radiocity', subsys,
                            promotes_inputs=['Q_Solar_In','r_1','F_I_1','F_I_BP','F_1_BP','F_BP_1','F','dL','B','T_BP'],
                            promotes_outputs=['*'])
        
        self.connect('T', 'radiocity.T_cav',src_indices=np.arange(disc_z+2,dtype=int),flat_src_indices=True)
        
        subsys = solid(disc_z = disc_z, disc_SiC = disc_SiC, disc_INS = disc_INS, disc_RPC = disc_RPC)
        self.add_subsystem('solid', subsys, 
                   promotes_inputs=['*'],
                   promotes_outputs=['*'])
        
        subsys = T_BP()
        self.add_subsystem('T_BP', subsys, 
                   promotes_inputs=['E','r_1'],
                   promotes_outputs=['*'])
        
        self.connect('Q','T_BP.Q',src_indices=(-1),flat_src_indices=True)
        
        subsys = T_corner(disc_SiC = disc_SiC)
        self.add_subsystem('T_corner', subsys,
                           promotes_inputs=['Ac_SiC'],
                           promotes_outputs=['*'])
        
        self.connect('T', 'T_corner.T_side',src_indices=(disc_z+2)*np.arange(1,disc_SiC+1,dtype=int),flat_src_indices=True)
        
        subsys = fluid(disc_z = disc_z, disc_RPC = disc_RPC)
        self.add_subsystem('fluid', subsys,
                           promotes_inputs=['h_','A_spec','V_RPC','m','cp','T_fluid_in'],
                           promotes_outputs=['*'])
        
        self.RPC_ind = np.zeros((disc_RPC*self.options['disc_z']),dtype=int)
        sic_2 = self.options['disc_SiC']+2
        z_2 = self.options['disc_z']+2
        i = sic_2*z_2+1
        i_fin = (sic_2+disc_RPC)*(z_2)-1
        
        k = 0
        while i<i_fin:
            
            if i%(z_2) == z_2-1:
                i=i+2
            else:
                self.RPC_ind[k] = i
                k=k+1
                i=i+1
        
        # np.save('RPC_ind',self.RPC_ind)
        
        self.connect('T', 'fluid.T_RPC', src_indices=self.RPC_ind, flat_src_indices=True)
        
        subsys = I_Rad(disc_z = disc_z, disc_SiC = disc_SiC, disc_INS = disc_INS, disc_RPC = disc_RPC)
        self.add_subsystem('I_Rad', subsys, 
                           promotes_inputs=['E','E_INS','omega','K_ex','r_n'],
                           promotes_outputs=['*'])
        
        self.connect('T', 'I_Rad.T_RPC', src_indices=self.RPC_ind, flat_src_indices=True)
        self.connect('T', 'I_Rad.T_RPC_CAV', flat_src_indices=True, src_indices=np.arange(disc_z)+(disc_SiC+1)*(disc_z+2)+1)
        self.connect('T', 'I_Rad.T_RPC_INS', flat_src_indices=True, src_indices=np.arange(disc_z)+(disc_SiC+disc_RPC+2)*(disc_z+2)+1)
        
        subsys = Tf_out(disc_RPC = disc_RPC)
        self.add_subsystem('Tf_out', subsys,
                           promotes_inputs=['m','cp','T_fluid_in','Q_Solar_In'],
                           promotes_outputs=['*'])
        
        self.Tfout_ind = np.arange(0,disc_RPC)*disc_z-1
        
        self.connect('T_fluid','Tf_out.T_fluid', src_indices=self.Tfout_ind, flat_src_indices=True)
        
        self.set_order(['init','T_BP','radiocity','solid','fluid','T_corner','I_Rad','Tf_out'])
        
        
        # self.nonlinear_solver =  om.NewtonSolver()
        # self.nonlinear_solver.options['solve_subsystems'] = True
        # # self.nonlinear_solver.linesearch = om.BroydenSolver()
        # self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        # self.nonlinear_solver.linesearch.options['maxiter'] = 10
        # self.nonlinear_solver.linesearch.options['iprint'] = 2
        # self.nonlinear_solver.linesearch.options['debug_print'] = True
        
        # self.nonlinear_solver = om.NonlinearRunOnce()
        
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['use_aitken'] = True
        self.nonlinear_solver.options['aitken_max_factor'] = 1.0
        
        # self.nonlinear_solver = om.NonlinearBlockJac()
        self.nonlinear_solver.options['err_on_non_converge'] = True
        
        self.nonlinear_solver.options['iprint'] = -1
        self.nonlinear_solver.options['maxiter'] = 1000
        rtol = 0.007;#print(rtol)
        atol = 1.0e-7
        self.nonlinear_solver.options['rtol'] = rtol
        self.nonlinear_solver.options['atol'] = atol
        
        self.linear_solver = om.ScipyKrylov()
        
        # print('Receiver.setup')
        
class ReceiverSubProblem(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('design',types=bool, default=False, desc='If True, design the receiver')
        self.options.declare('opt',types=bool, default=False, desc='If True, optimizes the receiver, in design mode')
        self.options.declare('tol_MaxT',types=float,default=0.02, desc='Max. temperature tolerance', upper=1, lower=0.00001)
        self.options.declare('T_Max', types=float, default=1000., desc='Max. temperature', upper=1500, lower=500)
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        self.options.declare('L', types=float, default=0.065, desc='Lenght of receiver for optimization')
        self.options.declare('s_RPC', types=float, default=0.015, desc='RPC thickness of receiver for optimization')
        self.options.declare('s_INS', types=float, default=0.1, desc='INS thickness of receiver for optimization')
    
    def setup(self):
        
        # This part for preliminary settings
        #optimization params
        s_RPC = 0.015
        s_INS = 0.1
        L = 0.065
        
        
        #optimizer params
        optimizer='SLSQP'
        opt_tol = 1e-3 #do not set less than 1e-5
        opt_maxiter = 100
        
        # design = self.options['design']
        opt = self.options['opt']
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        tol = self.options['tol_MaxT']
        T_Max = self.options['T_Max']
        
        #Common inputs and outputs
        #solar input
        self.add_input('Q_Solar_In',val=1000, desc='Total Solar Power Input',units='W')
        
        self.add_input('h_loss',val=15,desc='Heat transfer coefficient to ambient',units='W/(m**2*K)')
        
        #dim.s
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('s_SiC',val=0.005,desc='Thikness of SiC tube', units='m')
        
        self.add_output('eff', desc='Efficiency solar input to outlet')
        self.add_output('T_fluid_out', val=293.0, desc='Outlet temperature of air',units='K')
        
        # TODO: add outputs w.r.t. design and off-design cases!
        if not opt:
            self.add_input('s_RPC',val=s_RPC,desc='Thikness of RPC tube', units='m')
            self.add_input('s_INS',val=s_INS,desc='Thickness Insulation', units='m')
            self.add_input('L',val=L,desc='Length of the SiC tube', units='m')
        else:
            # self.add_output('s_RPC',val=s_RPC,desc='Thikness of RPC tube', units='m')
            # self.add_output('s_INS',val=s_INS,desc='Thickness Insulation', units='m')
            # self.add_output('L',val=L,desc='Length of the SiC tube', units='m')
            self.add_output('s_RPC',desc='Thikness of RPC tube', units='m')
            self.add_output('s_INS',desc='Thickness Insulation', units='m')
            self.add_output('L',desc='Length of the SiC tube', units='m')
        
        #mat.s
        self.add_input('E',val=0.9,desc='Emissivity')
        self.add_input('E_INS',val=0.5,desc='Emissivity of Insulation')
        self.add_input('omega', val=0.2, desc='scattering albedo')
        self.add_input('K_ex',val=200,desc='Extinction Coefficient',units='m**-1')
        self.add_input('k_INS',val=0.3,desc='Conductivity insulation',units='W/(m*K)')
        self.add_input('k_SiC',val=33,desc='Conductivity SiC',units='W/(m*K)')
        self.add_input('k_Air',val=0.08,desc='Conductivity air',units='W/(m*K)')
        self.add_input('A_spec',val=500,desc='Specific Surface of the RPC',units='m**-1')
        self.add_input('cp',val=1005.,desc='Specific Heat Capacity',units='J/(kg*K)')
        
        self.add_input('Mass_Flow', val=0.00068,desc='Mass flow rate', units='kg/s')
        
        # self.add_input('T_fluid_out_max', val=1000, units='degC', desc='Maximumt Outlet Fluid Temperature')
        # manuel setting for maximum temperature!..
        
        self.add_input('T_fluid_in', val=300.,desc='Inlet temperature of air',units='K')
        self.add_input('Tamb',val=293,desc='Ambient Temperature', units='K')
        
        self._problem = prob =  om.Problem()
        prob.model.add_subsystem('receiver', Receiver(disc_z = self.options['disc_z'],
                                                 disc_SiC=self.options['disc_SiC'],
                                                 disc_RPC=self.options['disc_RPC'],
                                                 disc_INS=self.options['disc_INS']), promotes=['*'])
        
        if opt:
            
            debug_print = ['desvars','objs','nl_cons','totals']
            # debug_print = ['desvars','objs','nl_cons']
            prob.driver = om.ScipyOptimizeDriver(debug_print = debug_print,
                                                 optimizer=optimizer, 
                                                 tol=opt_tol,
                                                 maxiter = opt_maxiter)
            
            prob.model.set_input_defaults('s_RPC', self.options['s_RPC'], units='m')
            prob.model.set_input_defaults('s_INS', self.options['s_INS'], units='m')
            prob.model.set_input_defaults('L', self.options['L'], units='m')
            
            # Efficiency optimization
            # prob.model.add_design_var('s_RPC',lower = 0.005, upper = 0.025, units='m', scaler= 0.40)
            # prob.model.add_design_var('s_INS',lower = 0.05,upper = 0.15, units='m', scaler= 0.01)
            # prob.model.add_design_var('L',lower = 0.02,upper = 0.07, units='m', scaler= 10)
            
            # prob.model.add_constraint('T_fluid_out', lower=T_Max, units='degC', scaler=0.001) # default 0.001 upper=(1+tol)*T_Max,
            # prob.model.add_constraint('T', upper=373, units='K',indices=((disc_z+2)*(3+disc_INS+disc_RPC+disc_SiC)+np.arange(disc_z+2,dtype=int)),flat_indices=True, scaler=0.001)
            # prob.model.add_constraint('Volume',lower=0.002, upper=0.003721609197258809, units='m**3' ,scaler=30)

            # prob.model.add_objective('eff_S2G',scaler=-1, adder=-1)#, adder=-1,scaler=100)#scaling should be increased
            
            # Volume optimization not working
            # prob.model.add_design_var('s_RPC',lower = 0.005, upper = 0.025, units='m', scaler= 1)
            # prob.model.add_design_var('s_INS',lower = 0.05,upper = 0.15, units='m', scaler= 1)
            # prob.model.add_design_var('L',lower = 0.02,upper = 0.07, units='m', scaler= 1)
            
            # prob.model.add_constraint('T_fluid_out', lower=T_Max*(1-tol/2),upper=T_Max*(1+tol/2), units='degC', scaler=0.001) # default 0.001 upper=(1+tol)*T_Max,
            # prob.model.add_constraint('T', upper=100,lower=80, units='degC',indices=((disc_z+2)*(3+disc_INS+disc_RPC+disc_SiC)+np.arange(disc_z+2,dtype=int)),flat_indices=True, scaler=0.001)
            # prob.model.add_objective('Volume', units='m**3', scaler=100)
            
            # Outlet Temperature Optimization
            prob.model.add_design_var('s_RPC',lower = 0.005, upper = 0.025, units='m', scaler= 0.40)
            prob.model.add_design_var('s_INS',lower = 0.05,upper = 0.15, units='m', scaler= 0.01)
            prob.model.add_design_var('L',lower = 0.02,upper = 0.07, units='m', scaler= 10)
            
            prob.model.add_objective('T_fluid_out', units='degC', scaler=-0.001) # default 0.001 upper=(1+tol)*T_Max,
            
            prob.model.add_constraint('T', upper=100,lower=80, units='degC',indices=((disc_z+2)*(3+disc_INS+disc_RPC+disc_SiC)+np.arange(disc_z+2,dtype=int)),flat_indices=True, scaler=0.001)
            prob.model.add_constraint('Volume',lower=0.002, upper=0.003721609197258809, units='m**3' ,scaler=30)

            
        self.declare_partials('*', '*',method='fd',step_calc='rel_avg')
        
        prob.setup()
    
    def compute(self, inputs, outputs):
        
        
        # design = self.options['design']
        opt = self.options['opt']
        prob = self._problem
        
        r_1 = inputs['r_1']
        s_SiC = inputs['s_SiC']
        
        prob.set_val('receiver.Q_Solar_In',inputs['Q_Solar_In'], units='W')
        prob.set_val('receiver.h_loss',inputs['h_loss'],units='W/(m**2*K)')
        prob.set_val('receiver.r_1',r_1 , units='m')
        prob.set_val('receiver.s_SiC', s_SiC, units='m')
        
        prob.set_val('receiver.E', inputs['E'])
        prob.set_val('receiver.E_INS', inputs['E_INS'])
        prob.set_val('receiver.omega', inputs['omega'])
        prob.set_val('receiver.K_ex', inputs['K_ex'], units='m**-1')
        prob.set_val('receiver.k_INS', inputs['k_INS'], units='W/(m*K)')
        prob.set_val('receiver.k_SiC', inputs['k_SiC'], units='W/(m*K)')
        prob.set_val('receiver.k_Air', inputs['k_Air'], units='W/(m*K)')
        prob.set_val('receiver.A_spec', inputs['A_spec'], units='m**-1')
        prob.set_val('receiver.cp', inputs['cp'], units='J/(kg*K)')
        
        prob.set_val('receiver.Mass_Flow',inputs['Mass_Flow'], units='kg/s')
        
        prob.set_val('receiver.T_fluid_in', inputs['T_fluid_in'], units='K')
        prob.set_val('receiver.Tamb', inputs['Tamb'], units='K')
        
        if not opt:
            
            prob.set_val('receiver.L', inputs['L'], units='m')
            prob.set_val('receiver.s_RPC', inputs['s_RPC'], units='m')
            prob.set_val('receiver.s_INS', inputs['s_INS'], units='m')
            
            prob.final_setup()
            prob.run_model()
        else:

            prob.final_setup()
            prob.run_driver()
            
            s_RPC = prob['s_RPC']
            outputs['s_RPC'] = s_RPC
            outputs['s_INS'] = prob['s_INS']
            outputs['L'] = prob['L']
            
            
            disc_z = self.options['disc_z']
            disc_SiC = self.options['disc_SiC']
            disc_RPC = self.options['disc_RPC']
            disc_INS = self.options['disc_INS']
            T = prob.get_val('receiver.T').reshape(disc_SiC+disc_RPC+disc_INS+4,disc_z+2)
            
            z_n = prob.get_val('receiver.z_n')
            r_n = prob.get_val('receiver.r_n')
            
            draw_contour(z_n[0,:], r_n[:,0], T-273, r_1+s_SiC, r_1+s_SiC+s_RPC, prob['receiver.Mass_Flow'], 10)
            
        outputs['eff'] = prob['eff_S2G']
        outputs['T_fluid_out'] =  prob['T_fluid_out']
        
    
if __name__ == "__main__":
    design=True
    opt = True
    
    p = om.Problem()
    
        
    p.model.add_subsystem('receiver', ReceiverSubProblem(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10,
                                                         design=design, opt=opt))
    
    p.setup()
    Mass_Flow=0.00066
    p.set_val('receiver.Mass_Flow', Mass_Flow, units='kg/s')
    
    
    p.set_val('receiver.Q_Solar_In', 1000., units='W')
    print('-'*20)
    print(f'Design:{design}')
    print(f'Optimization:{opt}')
    print('-'*20)
    p.run_model()
    
    p.model.list_outputs(units=True,prom_name=True,shape=False)
    p.model.list_inputs(units=True,prom_name=True,shape=False)
    
    # r_1 = p.get_val("receiver.r_1")
    # s_SiC = p.get_val("receiver.s_SiC")
    # s_RPC = p.get_val("receiver.s_RPC")
    # s_INS  = p.get_val("receiver.s_INS")
    
    # z_n = p["receiver.z_n"]#p.get_val('receiver.z_n')
    # r_n = p.get_val('receiver.r_n')
    # T = p.get_val('receiver.T').reshape(44,22)
    # draw_contour(z_n[0,:], r_n[:,0], T-273, r_1+s_SiC, r_1+s_SiC+s_RPC, p['receiver.Mass_Flow'], 10)
        
    # p.check_partials(compact_print=True, show_only_incorrect=True)
    
    """
    Mass_Flow = 0.00065
# i=10
# M=np.arange(-i,i)*0.000001+Mass_Flow

# for Mass_Flow in M:
    tic = time.time()
    p = om.Problem()
    p.model.add_subsystem('receiver', Receiver(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10))
    p.setup()    
    
    # this oart for initialize
    p.set_val('receiver.L', 0.065, units='m')
    
    r_1 = 0.015
    s_SiC = 0.005
    s_RPC = 0.015
    s_INS  = 0.1
    
    p.set_val('receiver.r_1', r_1, units='m')
    p.set_val('receiver.s_SiC',s_SiC, units='m')
    p.set_val('receiver.s_RPC',s_RPC, units='m')
    p.set_val('receiver.s_INS',s_INS, units='m')
    
    p.set_val('receiver.E', 0.9)
    #for P1
    p.set_val('receiver.E_INS', 0.5)
    p.set_val('receiver.omega', 0.2)
    
    p.set_val('receiver.h_loss', 15., units='W/(m**2*K)')
    
    p.set_val('receiver.k_INS', 0.3, units='W/(m*K)')
    p.set_val('receiver.k_SiC', 33., units='W/(m*K)')
    p.set_val('receiver.k_Air', 0.08, units='W/(m*K)')
    
    # p.set_val('receiver.porosity', 0.81)
    
    # p.set_val('receiver.p', 10., units='bar')
    
    # Mass_Flow = 0.00068
    p.set_val('receiver.Mass_Flow', Mass_Flow, units='kg/s')

    # p.set_val('receiver.D_nom', 0.00254, units='m')
    p.set_val('receiver.A_spec', 500.0, units='m**-1')
    p.set_val('receiver.K_ex', 200., units='m**-1')
    
    p.set_val('receiver.cp',1005.,units='J/(kg*K)')
    
    p.set_val('receiver.T_fluid_in',293, units='K')
    
    p.set_val('receiver.Tamb', 293., units='K')    
    
    # this part for radiocity
    p.set_val('receiver.Q_Solar_In', 1000., units='W')
    # p.set_val('receiver.sigma', 5.67*10**(-8), units='W/(m**2*K**4)')
    
    p.run_model()
    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)
    
    om.view_connections(p, outfile= "receiver.html", show_browser=False)
    om.n2(p, outfile="receiver_n2.html", show_browser=False)
    
    z_n = p.get_val('receiver.z_n')
    r_n = p.get_val('receiver.r_n')
    T = p.get_val('receiver.T').reshape(44,22)
    
    draw_contour(z_n[0,:], r_n[:,0], T-273, r_1+s_SiC, r_1+s_SiC+s_RPC, Mass_Flow, 10)
    
    Tf_out= p.get_val('receiver.T_fluid_out')
    if Tf_out<1273:
        print(f'failed : {Tf_out-273}')
    print(f'Mass_Flow: {Mass_Flow}')
    print(f'Tf_out: {Tf_out}K')
    # print('efficiency:',p.get_val('receiver.eff_S2G'))
    # Q = p.get_val('receiver.radiocity.Q')
    # print(Q)
    T = p.get_val('receiver.T')
    
    # print(T)
    # G =  np.load('G.npy')
    # data = p.check_partials(compact_print=True, show_only_incorrect=False, step_calc='rel_element', rel_err_tol=1e-6,abs_err_tol=1e-6)
    data = data['receiver.radiocity']
    # data_T = data['T_fluid','A_spec']['J_fd']
    data_V = data['Q','F_BP_1']['J_fd']
    # data_V = np.resize(data_V,(20,20))
    data_V_calc = data['Q','F_BP_1']['J_fwd']
    # data_V_calc = np.resize(data_V_calc,(20,20))
    
    err = data_V-data_V_calc
    
    max_=np.amax(err)
    min_=np.amin(err)
    
    # np.save('Tfluid.npy', p.get_val('receiver.T_fluid'))
    """