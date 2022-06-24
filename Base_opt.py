#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 00:22:36 2022

@author: tufanakba
MG3 integrated simple Brayton cycle,

receiver optimization problem inhereted from previous system problem solution
L, s_INS, s_RPC will be set from previous thermodynamic solution

"""

import openmdao.api as om
import pycycle.api as pyc
from py_Receiver import py_Receiver
from components.comp_map import AXI5
from components.MG3 import MG
from viewer import viewer, viewer2, map_plots, sankey
from receiver import ReceiverSubProblem

class Complex(pyc.Cycle):

    def initialize(self):
        # Initialize the model here by setting option variables such as a switch for design vs off-des cases
        self.options.declare('setting_mode', default='T4', values=['T4', 'nozzle'])

        super().initialize()
        
    def setup(self):
        
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())

        map_data = AXI5
        
        self.add_subsystem('comp', pyc.Compressor(map_data=map_data, map_extrap=True),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('rec', py_Receiver())
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269, map_extrap=True),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('MG', MG(), promotes_inputs=[('n','Nmech')])
        
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'])
        # self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=0))
        
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'rec.Fl_I')
        self.pyc_connect_flow('rec.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')
        
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')
        self.connect('shaft.trq_net', 'MG.trq')
        
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        
        # self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        # self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        # self.connect('inlet.F_ram', 'perf.ram_drag')
        # self.connect('nozz.Fg', 'perf.Fg_0')
        # self.connect('MG.P_el', 'perf.power')
        
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            
            
            balance.add_balance('turb_PR', val=5, lower=3, upper=8, eq_units='W', rhs_name='pwr_target')
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('MG.P_el', 'balance.lhs:turb_PR')
            
            balance.add_balance('W', lower=0.0001, upper=0.01, units='lbm/s', eq_units='degC', rhs_name='T4_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('rec.Fl_O:tot:T', 'balance.lhs:W')
            
            # balance.add_balance('comp_PR', val=1.5, lower=1.28, upper=6.43, eq_units='lbf', rhs_name='Fn_target')
            # self.connect('balance.comp_PR', 'comp.PR')
            # self.connect('perf.Fn', 'balance.lhs:comp_PR')
            
            # balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
            # self.connect('balance.W', 'inlet.Fl_I:stat:W')
            # self.connect('perf.Fn', 'balance.lhs:W')
            
            # balance.add_balance('comp_PR', val=2, lower=1.001, upper=8, eq_units='degC', rhs_name='T4_target')
            # self.connect('balance.comp_PR','comp.PR')
            # self.connect('rec.Fl_O:tot:T' ,'balance.lhs:comp_PR')
            
        else:
            
            
            balance.add_balance('Nmech', units='rpm', lower=7500., upper=17250., eq_units='W', rhs_name='pwr_target')
            self.connect('balance.Nmech', 'Nmech')
            self.connect('MG.P_el', 'balance.lhs:Nmech')
            
            if self.options['setting_mode'] == 'T4':
                # Nozzle alanı ile receiver outlet temperature'u korole etmek gerekecek! 1050 W'dan işlemek mantıklı!
                balance.add_balance('W', lower=0.0001, upper=0.01, units='lbm/s', eq_units='degC', rhs_val=1000,
                                    rhs_name='T4_target')
                # 1020'de çalışıyor
                self.connect('balance.W', 'inlet.Fl_I:stat:W')
                self.connect('rec.Fl_O:tot:T', 'balance.lhs:W')

            if self.options['setting_mode'] == 'nozzle':
                # balance.add_balance('W', lower=0.0001, upper=0.01, units='lbm/s', eq_units='inch**2')
                # self.connect('balance.W', 'inlet.Fl_I:stat:W')
                # self.connect('nozz.Throat:stat:area', 'balance.lhs:W')
                balance.add_balance('W', lower=0.0001, upper=0.01, units='lbm/s', eq_units=None, rhs_val=2.0)
                self.connect('balance.W', 'inlet.Fl_I:stat:W')
                self.connect('comp.map.RlineMap', 'balance.lhs:W')
                
        self.set_order(['fc', 'inlet', 'comp', 'rec', 'turb', 'nozz','shaft', 'MG', 'balance'])
        
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        
        newton.options['stall_limit'] = 3
        newton.options['stall_tol'] = 1e-4

        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options['iprint'] = 1
        newton.linesearch.options['maxiter'] = 5
        newton.linesearch.options['rho'] = 0.85  # work on impact of rho 0.85 for sample brayton

        self.linear_solver = om.DirectSolver()

        super().setup()

class MPComplex(pyc.MPCycle):

    def setup(self):
        # Create design instance of model
        self.pyc_add_pnt('DESIGN', Complex(design=True))

        self.set_input_defaults('DESIGN.Nmech', 15000.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.20)
        self.set_input_defaults('DESIGN.rec.MN', 0.55)
        self.set_input_defaults('DESIGN.turb.MN', 0.4)

        self.pyc_add_cycle_param('nozz.Cv', 0.99)
        
        self.od_MNs = 0.000001
        self.od_alts = 0.0
        
        self.od_pts = ['SelfSupplied','Motoring','Max_110', 'Min_75']
        self.od_pwrs = [0.,-50.,190., 100.]
        self.Q_Solar = [500.,400.,1100., 750.]
        setting_mode = ['nozzle','nozzle','T4', 'nozzle']
        
        
        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, Complex(design=False, setting_mode=setting_mode[i]))

            self.set_input_defaults(pt + '.fc.MN', val=self.od_MNs)
            self.set_input_defaults(pt + '.fc.alt', self.od_alts, units='ft')
            # self.set_input_defaults(pt+'.balance.Fn_target', self.od_Fns[i], units='lbf')
            self.set_input_defaults(pt + '.balance.pwr_target', self.od_pwrs[i], units='W')
            # self.set_input_defaults(pt+'.MG.P_el',-self.generator[i], units='W')
            self.set_input_defaults(pt + '.rec.Q_Solar_In', self.Q_Solar[i], units='W')
            # self.set_input_defaults(pt+'.comp.PR',5)
            # self.set_input_defaults(pt+'.turb.PR',5)

            # if setting_mode[i]=='nozzle':# If nozzzle balances mass flow
            #     self.connect('DESIGN.nozz.Throat:stat:area', pt+'.balance.rhs:W')
            
        self.pyc_connect_des_od('comp.s_PR', 'comp.s_PR')
        self.pyc_connect_des_od('comp.s_Wc', 'comp.s_Wc')
        self.pyc_connect_des_od('comp.s_eff', 'comp.s_eff')
        self.pyc_connect_des_od('comp.s_Nc', 'comp.s_Nc')

        self.pyc_connect_des_od('turb.s_PR', 'turb.s_PR')
        self.pyc_connect_des_od('turb.s_Wp', 'turb.s_Wp')
        self.pyc_connect_des_od('turb.s_eff', 'turb.s_eff')
        self.pyc_connect_des_od('turb.s_Np', 'turb.s_Np')

        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('comp.Fl_O:stat:area', 'comp.area')
        self.pyc_connect_des_od('rec.Fl_O:stat:area', 'rec.area')
        self.pyc_connect_des_od('turb.Fl_O:stat:area', 'turb.area')
        
        
        # self.pyc_connect_des_od('nozz.Throat:stat:area','balance.rhs:W')
        # self.pyc_use_default_des_od_conns()
            

if __name__ == "__main__":

    import time

    # Design conditions
    # fn = 0.01
    Q_Solar = 1000.
    T_4 = 1000.
    pwr_target = 170.
    # generator = 170.
    alt = 0.000001
    fc_MN = 1e-6

    comp_eff = 0.83
    turb_eff = 0.86
    
    # Design Param.s start
    
    s_RPC = 0.01
    s_INS = 0.1
    L = 0.035
    
    # Design Param.s final
    
    # L = 0.06644631131737304# m
    # s_INS = 0.08473123540881979# m 
    # s_RPC = 0.00876963671886617# m
    # # mass_flow = 0.0008255348001645682 #kg/s
    
    prob = om.Problem()
    mp_complex = prob.model = MPComplex()
    
    prob.setup(check=False)
    
    prob.set_val('DESIGN'+'.rec.rec.s_RPC', s_RPC, units='m')
    prob.set_val('DESIGN'+'.rec.rec.s_INS', s_INS, units='m')
    prob.set_val('DESIGN'+'.rec.rec.L', L, units='m')
        
    
    #Define the design point
    prob.set_val('DESIGN.fc.alt', alt, units='ft')
    prob.set_val('DESIGN.fc.MN', fc_MN)
    
    # prob.set_val('DESIGN.balance.Fn_target', fn, units='lbf')
    prob.set_val('DESIGN.balance.pwr_target', pwr_target, units='W')
    prob.set_val('DESIGN.balance.T4_target', T_4, units='degC') 
    prob.set_val('DESIGN.comp.PR', 5.2) #critical 2.15
    prob.set_val('DESIGN.comp.eff', comp_eff)
    prob.set_val('DESIGN.rec.Q_Solar_In', Q_Solar, units='W')
    prob.set_val('DESIGN.turb.eff', turb_eff)
    # prob.set_val('DESIGN.MG.P_el', -generator, units='W')
    
    
    # Set initial guesses for balances
    
    prob['DESIGN.balance.W'] = 0.00177#0.00122811 #147/100000
    prob['DESIGN.balance.turb_PR'] = 4.463
    # prob['DESIGN.balance.comp_PR'] = 4.489
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.670 #518.665288153
    
    for i,pt in enumerate(mp_complex.od_pts):

        # initial guesses
        
        prob[pt+'.balance.W'] = 0.00177
        prob[pt+'.balance.Nmech'] = 15000
        prob[pt+'.fc.balance.Pt'] = 14.6955113159
        prob[pt+'.fc.balance.Tt'] = 518.670                         
        prob[pt+'.turb.PR'] = 4.5
        prob[pt+'.comp.PR'] = 4.5
        prob[pt+'.comp.map.RlineMap'] = 2.0
    
        prob.set_val(pt+'.rec.rec.s_RPC', s_RPC, units='m')
        prob.set_val(pt+'.rec.rec.s_INS', s_INS, units='m')
        prob.set_val(pt+'.rec.rec.L', L, units='m')
        
    
    
    
    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    
    prob.run_model()
    print("time", time.time() - st)
    st = time.time()
    
    w_ini = mp_complex.get_val('DESIGN.inlet.Fl_O:stat:W', units='kg/s')
    T_fluid_in = mp_complex.get_val('DESIGN.rec.rec.T_fluid_in', units='K')
    
    diff = 1
    limit = 0.001
    
    sankey(prob,'DESIGN')
    
    L_log=[]
    s_INS_log=[]
    s_RPC_log=[]
    m_dot_log=[]
    diff_log=[]
    
    m_dot=float(w_ini)
    
    L_log.append(L)
    s_INS_log.append(s_INS)
    s_RPC_log.append(s_RPC)
    m_dot_log.append(m_dot)
    
    prob.cleanup()

    while diff>limit:
        
        p = om.Problem()
            
        p.model.add_subsystem('receiver', ReceiverSubProblem(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10,
                                                              design=True, opt=True, T_Max=T_4,
                                                              L=L, s_INS=s_INS, s_RPC=s_RPC))
                                                              
        p.setup()
        # TODO: You shoul set the DESIGN vals in prob to p variables
        p.set_val('receiver.Q_Solar_In', Q_Solar, units='W')
        p.set_val('receiver.Mass_Flow', m_dot, units='kg/s')
        p.set_val('receiver.T_fluid_in',T_fluid_in, units='K')
        
        p.final_setup()
        p.run_model()
        
        L = float(p.get_val('receiver.L', units='m'))
        s_INS=float(p.get_val('receiver.s_INS', units='m'))
        s_RPC=float(p.get_val('receiver.s_RPC', units='m'))
        
        L_log.append(L)
        s_INS_log.append(s_INS)
        s_RPC_log.append(s_RPC)
        
        print("time", time.time() - st)
        st = time.time()
        p.cleanup()
        
        prob = om.Problem()
        mp_complex = prob.model = MPComplex()
        prob.setup(check=False)
        
        prob.set_val('DESIGN.rec.rec.s_RPC', s_RPC, units='m')
        prob.set_val('DESIGN.rec.rec.s_INS', s_INS, units='m')
        prob.set_val('DESIGN.rec.rec.L', L, units='m')
        
        #Define the design point
        prob.set_val('DESIGN.fc.alt', alt, units='ft')
        prob.set_val('DESIGN.fc.MN', fc_MN)
        
        prob.set_val('DESIGN.comp.PR', 5.2) #critical 2.15
        prob.set_val('DESIGN.balance.pwr_target', pwr_target, units='W')
        prob.set_val('DESIGN.balance.T4_target', T_4, units='degC') 
        # prob.set_val('DESIGN.comp.PR', 5) #critical 2.15
        prob.set_val('DESIGN.comp.eff', comp_eff)
        prob.set_val('DESIGN.rec.Q_Solar_In', Q_Solar, units='W')
        prob.set_val('DESIGN.turb.eff', turb_eff)

        # Set initial guesses for balances
        # prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 0.00177#0.00122811 #147/100000
        prob['DESIGN.balance.turb_PR'] = 4.463
        # prob['DESIGN.balance.comp_PR'] = 5
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.670 #518.665288153
        
        for i,pt in enumerate(mp_complex.od_pts):

            # initial guesses
            prob[pt+'.balance.W'] = 0.00177
            
            prob[pt+'.balance.Nmech'] = 15000
            prob[pt+'.fc.balance.Pt'] = 14.6955113159
            prob[pt+'.fc.balance.Tt'] = 518.670                         
            prob[pt+'.turb.PR'] = 4.5
            prob[pt+'.comp.PR'] = 4.5
            prob[pt+'.comp.map.RlineMap'] = 2.0
            
            # TODO: check this line
            prob.set_val(pt+'.rec.rec.s_RPC', s_RPC, units='m')
            prob.set_val(pt+'.rec.rec.s_INS', s_INS, units='m')
            prob.set_val(pt+'.rec.rec.L', L, units='m')
        
        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

        prob.run_model()
        
        m_dot=float(mp_complex.get_val('DESIGN.inlet.Fl_O:stat:W', units='kg/s'))
        m_dot_log.append(m_dot)
        
        print("time", time.time() - st)
        st = time.time()
        
        sankey(prob,'DESIGN')
        viewer(prob, 'DESIGN')
        viewer2(prob, 'DESIGN')
        w = m_dot
        
        diff=abs(1-w/w_ini)
        w_ini=w
        
        diff_log.append(diff)

        print(f"L = {float(prob.get_val('DESIGN.rec.rec.L', units='m'))} m")
        print(f"INS = {float(prob.get_val('DESIGN.rec.rec.s_INS', units='m'))} m ")
        print(f"RPC = {float(prob.get_val('DESIGN.rec.rec.s_RPC', units='m'))} m")
        print(f"mass_flow = {w} kg/s")
        
        if diff>limit:
            prob.cleanup()
        
        
    for pt in ['DESIGN']+mp_complex.od_pts:
        viewer(prob, pt)
        viewer2(prob, pt)
        map_plots(prob, pt)
    
    print()
    print("time", time.time() - st)
    
    print('shaft power [W]:  ')
    print(mp_complex.get_val('DESIGN.shaft.pwr_net', units='W'))
    print('receiver outlet temperature  [oC]:')
    print(mp_complex.get_val('DESIGN.rec.Fl_O:tot:T', units='degC'))
    print('mass flow rate [kg/s]:')
    print(mp_complex.get_val('DESIGN.inlet.Fl_O:stat:W', units='kg/s'))
