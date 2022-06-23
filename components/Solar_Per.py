#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:24:12 2022

@author: tufanakba
Solar Performance Model

first efficiency of receiver will be impemented
second overall system efficiency...
"""

from openmdao.api import ExplicitComponent, Problem, IndepVarComp


class Performance(ExplicitComponent):
    """Component to calculate overall engine performance parameters"""
    
    def initialize(self):

        self.add_input('Q_Solar_In', 1000, units='W', desc='Solar heat input')
    
    def setup(self):
        
        pass
    
    def compute(self, inputs, outputs):
        
        pass
    
    def compute_partials(self, inputs, partials):
        
        pass
    
if __name__ == "__main__":
    
    pass