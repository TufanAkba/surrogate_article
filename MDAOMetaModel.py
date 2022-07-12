#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:23:26 2022

@author: tufanakba
Meta model test for problem solving
"""

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    
    
    p = om.Problem()
    
    #get train data
    folder = '4_5'
    vol = np.loadtxt(folder + '/vol.csv')
    Tfo = np.loadtxt(folder + '/Tfo.csv')
    Tfi = np.loadtxt(folder + '/Tfi.csv')
    T_o = np.loadtxt(folder + '/T_o.csv')
    rpc = np.loadtxt(folder + '/rpc.csv')
    m_d = np.loadtxt(folder + '/m_dot.csv')
    L = np.loadtxt(folder + '/L.csv')
    ins = np.loadtxt(folder + '/ins.csv')
    
    surr = om.MetaModelUnStructuredComp(default_surrogate=om.ResponseSurface())
    # surr = om.MetaModelUnStructuredComp(default_surrogate=om.NearestNeighbor(interpolant_type='linear'))
    # surr = om.MetaModelUnStructuredComp(default_surrogate=om.NearestNeighbor(interpolant_type='rbf', num_neighbors=5, rbf_family=-2))
    # surr = om.MetaModelUnStructuredComp(default_surrogate=om.NearestNeighbor(interpolant_type='weighted'))
    # surr = om.MetaModelUnStructuredComp(default_surrogate=om.KrigingSurrogate(eval_rmse= True))
    
    surr.add_input('Tfi',training_data = Tfi)
    surr.add_input('m_dot',training_data = m_d)
    surr.add_input('rpc',training_data=rpc)
    surr.add_input('ins',training_data=ins)
    surr.add_input('L',training_data=L)
    surr.add_input('T_o',training_data=T_o)
    surr.add_input('vol', training_data=vol)
    surr.add_output('Tfo', training_data=Tfo)
    
    p.model.add_subsystem('surr', surr, promotes=['*'])
    p.setup()
    
    #get validation data
    folder = 'valid'
    vol = np.loadtxt(folder + '/vol.csv')
    Tfo = np.loadtxt(folder + '/Tfo.csv')
    Tfi = np.loadtxt(folder + '/Tfi.csv')
    T_o = np.loadtxt(folder + '/T_o.csv')
    rpc = np.loadtxt(folder + '/rpc.csv')
    m_d = np.loadtxt(folder + '/m_dot.csv')
    L = np.loadtxt(folder + '/L.csv')
    ins = np.loadtxt(folder + '/ins.csv')
    
    # dataset = pd.read_csv('df.csv')
    # print(dataset.columns) # get column names
    # inputs=dataset[['rpc','Tfi','m_dot','ins','L']]
    # outputs=dataset[['Tfo','vol','T_o']]
    # 
    # sns.pairplot(dataset, diag_kind='kde')
    # sns.pairplot(inputs, diag_kind='kde')
    # sns.pairplot(outputs, diag_kind='kde')
    
    
    err=[]
    Tfo_surr = []

    for i in range(vol.size):
        p.set_val('vol',vol[i])
        p.set_val('Tfi',Tfi[i])
        p.set_val('T_o',T_o[i])
        p.set_val('rpc',rpc[i])
        p.set_val('m_dot',m_d[i])
        p.set_val('L',L[i])
        p.set_val('ins',ins[i])
        
        p.final_setup()
        p.run_model()
        
        Tfo_surr.append(float(p.get_val('Tfo')))


    err=Tfo-Tfo_surr
    
    err=(1-Tfo_surr/Tfo)*100
    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.hist(err, bins=25)
    plt.xlabel('Prediction Error [%]')
    _ = plt.ylabel('Count')
    
    min_lim = min(np.amin(Tfo_surr),np.amin(Tfo))
    max_lim = max(np.amax(Tfo_surr),np.amax(Tfo))
    

    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.axes(aspect='equal')
    plt.scatter(Tfo, Tfo_surr)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [min_lim, max_lim]
    plt.xlim(lims)
    plt.ylim(lims)
    
    RMS_text = "{:.2%}".format((np.corrcoef(Tfo, Tfo_surr)[0,1])**2)
    RMSE = np.sqrt(np.square(np.subtract(Tfo,Tfo_surr)).mean())
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1.05*min_lim, 0.95*max_lim, r'$R^2$' + f' = {RMS_text}',bbox=props)
    
    _ = plt.plot(lims, lims)
    
    