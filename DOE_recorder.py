#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 01:03:31 2022

@author: tufanakba
recorder code added optimizer

This code is written for creating surrogate model training data
Generates recorder file (cases.sql)
"""

import openmdao.api as om
import time
import numpy as np

from receiver import Receiver

if __name__ == '__main__':
    # TODO: set the number of sampling.
    # In the article 10**4 used for training and 10**2 for validation
    samples = 10 ** 4

    t0 = time.time()
    disc_z = 20
    disc_SiC = 10
    disc_RPC = 20
    disc_INS = 10

    # design=True
    # opt = True

    p = om.Problem()

    p.model.add_subsystem('receiver', Receiver(disc_z=disc_z, disc_SiC=disc_SiC, disc_RPC=disc_RPC, disc_INS=disc_INS))
    debug_print = ['desvars', 'objs', 'nl_cons', 'totals']

    # p.driver = om.DOEDriver(om.UniformGenerator(num_samples=1000),debug_print = debug_print)
    p.driver = om.DOEDriver(om.LatinHypercubeGenerator(samples=samples), debug_print=debug_print)

    s_RPC = 0.005
    s_INS = 0.05
    L = 0.02
    Mass_Flow = 0.0002
    T_inlet = 400

    p.model.set_input_defaults('receiver.Mass_Flow', Mass_Flow, units='kg/s')
    p.model.set_input_defaults('receiver.s_RPC', s_RPC, units='m')
    p.model.set_input_defaults('receiver.s_INS', s_INS, units='m')
    p.model.set_input_defaults('receiver.L', L, units='m')
    p.model.set_input_defaults('receiver.T_fluid_in', T_inlet, units='K')

    p.model.add_design_var('receiver.s_RPC', lower=0.005, upper=0.025, units='m')
    p.model.add_design_var('receiver.s_INS', lower=0.05, upper=0.25, units='m')
    p.model.add_design_var('receiver.L', lower=0.02, upper=0.1, units='m')
    p.model.add_design_var('receiver.Mass_Flow', lower=0.0002, upper=0.001, units='kg/s')
    p.model.add_design_var('receiver.T_fluid_in', lower=300, upper=600, units='K')

    p.model.add_objective('receiver.T_fluid_out', units='degC', scaler=-1)

    p.model.add_constraint('receiver.T', upper=100, lower=80, units='degC', indices=(
                (disc_z + 2) * (3 + disc_INS + disc_RPC + disc_SiC) + np.arange(disc_z + 2, dtype=int)),
                           flat_indices=True)
    p.model.add_constraint('receiver.Volume', lower=0.002, upper=0.003721609197258809, units='m**3')

    # p.model.add_objective('receiver.eff_S2G',scaler=scaler, adder=0)#, adder=-1,scaler=100)#scaling should be increased

    # Create a recorder
    recorder = om.SqliteRecorder('cases.sql')

    # Attach recorder to the driver
    p.driver.add_recorder(recorder)

    p.setup()

    # Mass_Flow=0.00066
    # p.set_val('receiver.Mass_Flow', Mass_Flow, units='kg/s')
    p.set_val('receiver.Q_Solar_In', 1000., units='W')

    # print('-'*20)
    # print(f'Design:{design}')
    # print(f'Optimization:{opt}')
    # print('-'*20)

    # p.run_model()
    p.run_driver()

    # p.model.list_outputs(units=True,prom_name=True,shape=False)
    # p.model.list_inputs(units=True,prom_name=True,shape=False)

    print('Elapsed time is', time.time() - t0, 'seconds', sep=None)

    # p.check_partials(compact_print=True)
