#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:50:35 2020

Post process of the P1_model receiver code

@author: tufanakba
"""


import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime

def draw_contour(z_n, r_n, T, r1, r2, m_dot, countour_number):
    
    origin = 'lower'
    
    X, Y = np.meshgrid(z_n, r_n)
    fig1, ax2 = plt.subplots(figsize=(10,8),constrained_layout=True)

    #for color
    # CS = ax2.contourf(X, Y, T, countour_number, origin=origin) # cmap=plt.cm.bone,
    
    # for value on grapgh
    CS = ax2.contour(X, Y, T, countour_number, origin=origin) # cmap=plt.cm.bone,
    # this is for showing values on the figure
    ax2.clabel(CS, inline=1, fontsize=8)
    
    
    # print(z_n.shape); #for matrix check!
    ax2.plot((z_n[0],z_n[-1]),[r1,r1],'-k')
    ax2.plot((z_n[0],z_n[-1]),[r2,r2],'-k')
    
    ax2.set_title('Temperature Distribution (oC) '+'m_dot='+str(abs(m_dot))+'kg/s')
    ax2.set_xlabel('Z Dir. [m]')
    ax2.set_ylabel('R Dir. [m]')
    
    
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel('Temperature [oC]')
    
   
    
    # Records as the timestamp
    # now=datetime.now()
    # plt.savefig(now.strftime("%Y_%m_%d_%H_%M_%S.png"), dpi=fig1.dpi)
    
    #plots the figure
    plt.show()