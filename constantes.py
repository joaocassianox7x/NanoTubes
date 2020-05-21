#Constantes

import numpy as np
import numpy.linalg as alg
import time
import multiprocessing as mp
import numba
#import matplotlib.pyplot as plt

pool=mp.Pool(mp.cpu_count())
ti=time.time()
#--------------------
#CONSTANTES NUMERICAS
ngridy=np.int64(4)
nsite=np.int64(2)
norbit=np.int64(8)
cp=16 #casas de precisao
Einicial=0 #campo inicial
Efinal=1.5 #campo final
campos=np.linspace(Einicial,Efinal,2*mp.cpu_count()) #lista de campos
#-------------------
#CONSTANTES FISICAS
lso=np.float64(0.1)
lsofact=np.float64(1/(3*np.sqrt(3)))
v1=np.int64(1)
ll=np.float64(0.23)
ass=np.float64(3.86/np.sqrt(3))
Ra=np.float64(3*nsite*ass/(2*np.pi))
Rb=np.float64(Ra-2*ll)
#-------------------
#CONSTANTES PARA GREEN
bandwidth=np.int64(7)
enerinit=np.float64(-3.5)
energrid=np.int64(5*10**1)
eta=1j*np.float64(5*bandwidth/energrid)
#-------------------