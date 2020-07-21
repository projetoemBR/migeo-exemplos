#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 27 23:23:20 2019

MT3D forward modeling with 1D geoelectric model

@author: diego
"""
import sys
import time
import inspect
#from SimPEG import (
#    Mesh, Utils, Maps, DataMisfit, Regularization, Optimization,
#    InvProblem, Directives, Inversion, Versions
#)
#from SimPEG.Utils import plotLayer
import SimPEG as simpeg
from SimPEG.EM import NSEM
from scipy.constants import mu_0

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver

from scipy.constants import mu_0, epsilon_0 as eps_0
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz


nFreq = 40
freqs = np.logspace(-2, 2, nFreq)

#  2D MODEL #######################################################


dx = 50#100    # minimum cell width (base mesh cell width) in x
dy = 50    # minimum cell width (base mesh cell width) in y
dz = 50#100    # minimum cell width (base mesh cell width) in z

x_length = 40000 #34316 #90000.     # domain width in x
y_length = 1000 #34316 #90000.     # domain width in y
z_length = 60000 #16584  #90000.     # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))


# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
M = simpeg.Mesh.TreeMesh([hx, hy, hz], x0='CCC')


# # Refine surface topography
# [xx, yy] = np.meshgrid(M.vectorNx, M.vectorNy)
# [xx, yy,zz] = np.meshgrid([-4000,4000], [-1000,1000],[-100,100])
# zz = 0.*(xx**2 + yy**2)  - 0.
# #zz = np.zeros([300,300])
# pts = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]
# M = refine_tree_xyz(
#     M, pts, octree_levels=[2, 2], method='surface', finalize=False
#     )

# Refine box
xp, yp, zp = np.meshgrid([-50000., 50000.], [-10000., 10000.], [-1000., -1100.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

M = refine_tree_xyz(
    M, xyz, octree_levels=[1,1], method='box', finalize=False)  

# #Refine rest of the grid
# def refine(cell):
#     if np.sqrt(((np.r_[cell.center]-0.5)**2).sum()) < 4000:
#         return 1
#     return 1


# M.refine(refine)



M.finalize()

print(M.nC)


# DEFININDO MODELO 3D
conds = [1e-2,1e0]   # [heterogeneidade,background]
sig = simpeg.Utils.ModelBuilder.defineBlock(
    M.gridCC, [-20000, -10000, -1000], [20000, 10000, -1100], conds)
    
sig[M.gridCC[:,2] > 0] = 1e-18

sigBG = np.zeros(M.nC) + conds[1]
sigBG[M.gridCC[:, 2] > 0] = 1e-18   


# MESH 1D (para modelo de background)
mesh1d = simpeg.Mesh.TensorMesh([M.hz], np.array([M.x0[2]]))

sigBG1d = np.zeros(mesh1d.nC) + conds[1]
sigBG1d[mesh1d.gridCC > 0] = 1e-18
   

########################################################


fig,axes = plt.subplots(num=1,clear=True)
M.plotSlice(np.log(sig), grid=True, normal='y',ax=axes)



# interrupt code  after build mesh
cont_flag = 0

if cont_flag == 1:
    sys.exit()


##-=========================================================================================================
## %% Setup the layout of the survey, set the sources and the connected receivers
# Receiver locations
#    rx_x, rx_y = np.meshgrid(np.arange(-600, 601, 100), np.arange(-600, 601, 100))
rx_x = np.array([0.])
rx_y = np.array([0.])
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2),  np.zeros((np.prod(rx_x.shape), 1))))



# Receivers
rxList = []
for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
    rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'real'))
    rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'imag'))
for rx_orientation in ['zx', 'zy']:
    rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'real'))
    rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'imag'))

# Source list,
srcList = []
for freq in freqs:
    #srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq))
    srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq, sigBG1d, sigBG))
# Make the survey
survey = NSEM.Survey(srcList)
#survey.mtrue = m_true

# Set the problem
problem = NSEM.Problem3D_ePrimSec(M, sigma=sig, sigmaPrimary=sigBG)
problem.pair(survey)
problem.Solver = Solver

# Calculate the data
fields = problem.fields()    # returns secondary field


# ## ADICIONADO POR DIEGO EM 01 MAR 2020 --------------------------------------

grid_field_px = np.empty((M.nE,nFreq),dtype=complex)
grid_field_py = np.empty((M.nE,nFreq),dtype=complex)
for i in range(nFreq):
    grid_field_px[:,i] = np.transpose(fields._getField('e_pxSolution', i))
    grid_field_py[:,i] = np.transpose(fields._getField('e_pySolution', i))


# campos E e H caluclado em todas as arestas d malha
e_px_full  = fields._e_px(grid_field_px, srcList)
e_py_full  = fields._e_py(grid_field_py, srcList)
h_px_full  = fields._b_px(grid_field_px, srcList)/mu_0
h_py_full  = fields._b_py(grid_field_py, srcList)/mu_0



ex_px_field = e_px_full[0:np.size(M.gridEx,0),:]
ex_py_field = e_py_full[0:np.size(M.gridEx,0),:]


# é necessário interpolar o campo h nas arestas
Pbx = M.getInterpolationMat(M.gridEx, 'Fx')
hx_px_field = Pbx*h_px_full
hx_py_field = Pbx*h_py_full

Zij_all_edgesx = -ex_px_field/hx_py_field


Pz = M.getInterpolationMat([0.,0.,0.],'Ex')
Pz = Pz[0,0:np.size(M.gridEx,0)]
Zij = np.empty((nFreq),dtype=complex)
for i in range(np.shape(Zij_all_edgesx)[1]):
    Zij[i] = Pz*Zij_all_edgesx[:,i]



rho_app = 1/(2*np.pi*freqs*mu_0) * abs(Zij)**2


phs_simpeg     = np.arctan2(Zij.imag, Zij.real)*(180./np.pi)




# ANALYTIC SOLUTION MT1D #############################################################        
        
mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
resistivities = np.array([1.,100.,1.]) #[100,10,200] #[300, 2500, 0.8, 3000, 2500];

thicknesses = [1000.,100] #[300,300] #[200, 400, 40, 500];
n = len(resistivities);

#nFreq = 13
frequencies = freqs #np.logspace(1, -2, nFreq) #[1,5,10,25,50,100] #data[:,0] ##[0.0001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,10000];

rhoapp      = np.zeros(len(frequencies))
phs         = np.zeros(len(frequencies))

#print('freq\tares\t\t\tphase');
for i,frequency in enumerate(frequencies):   
    w =  2*math.pi*frequency;       
    impedances = list(range(n));
    #compute basement impedance
    impedances[n-1] = cmath.sqrt(w*mu*resistivities[n-1]*1j);
   
    for j in range(n-2,-1,-1):
        resistivity = resistivities[j];
        thickness = thicknesses[j];
  
        # 3. Compute apparent resistivity from top layer impedance
        #Step 2. Iterate from bottom layer to top(not the basement) 
        # Step 2.1 Calculate the intrinsic impedance of current layer
        dj = cmath.sqrt((w * mu * (1.0/resistivity))*1j);
        wj = dj * resistivity;
        # Step 2.2 Calculate Exponential factor from intrinsic impedance
        ej = cmath.exp(-2*thickness*dj);                     
    
        # Step 2.3 Calculate reflection coeficient using current layer
        #          intrinsic impedance and the below layer impedance
        belowImpedance = impedances[j + 1];
        rj = (wj - belowImpedance)/(wj + belowImpedance);
        re = rj*ej; 
        Zj = wj * ((1 - re)/(1 + re));
        impedances[j] = Zj;    

     # Step 3. Compute apparent resistivity from top layer impedance
    Z = impedances[0];
    absZ = abs(Z);
    apparentResistivity = (absZ * absZ)/(mu * w);
    rhoapp[i] = apparentResistivity
    phase = math.atan2(Z.imag, Z.real)*(180./np.pi)
    phs[i] = phase
   #    print(frequency, '\t', apparentResistivity, '\t', phase);




# PLOT RESULTS --------------------------------------------------------------------         
    
plt.style.use('ggplot')    

# plot mcsem results hc x no hc
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(frequencies, rhoapp, 'o-k',frequencies, rho_app, '--.r')
ax1.set_xlabel('Frequência (Hz)')
ax1.set_ylabel('Resistividade Aparente (Rho.m)')
ax1.invert_xaxis()
ax1.set_xscale('log')
ax1.grid('True')
ax1.legend(('Analytic','Numeric'))    # model1 = no carb,  model2 = carb

ax2.plot(frequencies, phs, 'o-k', frequencies, phs_simpeg, '--.r')
ax2.set_xlabel('Frequência (Hz)')
ax2.set_ylabel('Fase (graus)')
ax2.invert_xaxis()
ax2.set_xscale('log')

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.legend(('Analytic','Numeric'))
ax2.grid('True')

fig.tight_layout()
plt.show()

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
