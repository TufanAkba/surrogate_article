#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:41:49 2022

@author: tufanakba
driver_postprocess

This file reads the sql file generated from DOE_recorder file and saves as .csv
"""

import openmdao.api as om
import numpy as np

# Instantiate your CaseReader
# TODO: update recorder below!
cr = om.CaseReader("cases.sql")
driver_cases = cr.list_cases('driver', recurse=False)


rpc_values = []
ins_values = []
L =[]
m_dot = []
T_inlet = []

volume = []
T_outer = []
T_fluid_out = []

for case_id in driver_cases:
    case = cr.get_case(case_id)
    design_vars = case.get_design_vars(scaled=False)
    objectives = case.get_objectives(scaled=False)
    constraints = case.get_constraints(scaled=False)
    
    rpc_values.append(float(design_vars['receiver.s_RPC']))
    ins_values.append(float(design_vars['receiver.s_INS']))
    L.append(float(design_vars['receiver.L']))
    m_dot.append(float(design_vars['receiver.Mass_Flow']))
    T_inlet.append(float(design_vars['receiver.T_fluid_in']))

    volume.append(float(constraints['receiver.Volume']))
    T_outer.append(float(np.amax(constraints['receiver.T'])))
    # T_outer.append(constraints['receiver.T'])
    T_fluid_out.append(float(objectives['receiver.T_fluid_out']))

np.savetxt('ins.csv',ins_values,delimiter=',')
np.savetxt('L.csv',L,delimiter=',')
np.savetxt('m_dot.csv',m_dot,delimiter=',')
np.savetxt('rpc.csv',rpc_values,delimiter=',')
np.savetxt('Tfo.csv',T_fluid_out,delimiter=',')
np.savetxt('Tfi.csv',T_inlet,delimiter=',')
np.savetxt('T_o.csv',T_outer,delimiter=',')
np.savetxt('vol.csv',volume,delimiter=',')