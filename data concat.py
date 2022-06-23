#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:09:40 2022

@author: tufanakba
for pairplot all csv files combined as a single file
Please locate the test data file
"""


import pandas as pd

import glob


if __name__ == "__main__":
    
    # TODO: locate the test data folder
    folder = 'data'
    
    all_files = glob.glob(folder + "/*.csv")
    df=pd.DataFrame()
    for f in all_files:
        
        title=f.split('.')[0]
        data=pd.read_csv(f,header=None)
        df[title] = data
    
    df.to_csv('df.csv', sep='\t', encoding='utf-8')
    
    
    
    
    
    