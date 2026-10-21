# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:14:37 2020

@author: medisp-2
"""
import time
import sys
def cls():
    print(chr(27) + "[2J") 
def pause():
    input("PRESS ENTER TO CONTINUE.")
    #    ------------------------------------------------------------
def tic():
    t1=float(time.time());
    return (t1)
#------------------------------------------------------------
def toc(t1,s):
    t2=float(time.time());dt=t2-t1;
    s1='time taken '+s 
    print("%s %e" % (s1,dt) )     
#---------------------------------------------------------
def RETURN():
    sys.exit()
     
