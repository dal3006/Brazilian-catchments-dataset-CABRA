import pandas as pd
import numpy as np
import os
import pymannkendall as mk
import pickle 

from utils import load_forcing, basin_area, load_discharge,get_basin_id

### Define initial and final timestamp 
inicial='1981-01-01'
last='2009-12-31'

### Root path

path='/home/pedrozamboni/Documentos/doutorado/dataset/cabra/v1'
basin_code=10
basin_idx=get_basin_id(path)

results={}
for i in basin_idx:

	### Load data
	forcing=load_forcing(path,basin_code).loc[inicial:last]
	discharge=load_discharge(path,basin_code).loc[inicial:last]

	lista=[]
	### Trend for forcing variables
	for elem in forcing.iloc[:,3:].columns.values.tolist():
		trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(forcing[elem].loc[inicial:last])
		#print(elem,trend, h, p, z, Tau, s, var_s, slope, intercept)
		lista.append((elem,trend, h, p, z, Tau, s, var_s, slope, intercept))

	
	### Trend for dischrage
	trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(discharge.iloc[:,3:4].loc[inicial:last]) 
	#print('discharge',trend, h, p, z, Tau, s, var_s, slope)'''
	lista.append((elem,trend, h, p, z, Tau, s, var_s, slope, intercept))

	results[i]=lista
###fazer o for para fazer pra todas e fazer algo pra salvar###
	### Load data

pickle.dump(results, open( "results.pkl", "wb" ))
