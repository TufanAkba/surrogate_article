#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 22:20:41 2022

@author: tufanakba
surrogate model integration to Brayton cycle model
"""

import openmdao.api as om
# import pycycle.api as pyc
# from py_Receiver import py_Receiver
# from components.comp_map import AXI5
# from components.MG3 import MG
from viewer import viewer, viewer2, map_plots, sankey
# from receiver import ReceiverSubProblem
from MDAOMetaModel_Opt import surrOpt

from MPComplex import MPComplex

import time

if __name__ == "__main__":
    
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
            
        p.model.add_subsystem('receiver', 
                              surrOpt(L=L, s_INS=s_INS, s_RPC=s_RPC)
                              ,promotes=['*'])
                                                              
        p.setup()
        # TODO: You shoul set the DESIGN vals in prob to p variables
        # p.set_val('receiver.Q_Solar_In', Q_Solar, units='W')
        p.set_val('m_dot', m_dot,units='kg/s')
        p.set_val('Tfi',T_fluid_in, units='K')
        
        p.final_setup()
        p.run_model()
        
        L = float(p.get_val('receiver.L', units='m'))
        s_INS=float(p.get_val('receiver.ins', units='m'))
        s_RPC=float(p.get_val('receiver.rpc', units='m'))
        
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
