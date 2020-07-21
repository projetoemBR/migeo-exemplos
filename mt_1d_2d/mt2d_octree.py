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





## Open and read file ( mt2d finite element)
infile = open('xs.txt', 'r')
xs = [float(line) for line in infile.readlines()]
infile.close()


infile = open('rho_ap.txt', 'r')
rho_ap = [float(line) for line in infile.readlines()]
infile.close()





# MAIN PROGRAM ---------------------------------------------------------------

t= time.time()

# def run(plotIt=True):

nFreq = 1
freqs = np.logspace(-2, 2, nFreq)

    
##########



#  2D MODEL #######################################################




dx = 50#100    # minimum cell width (base mesh cell width) in x
dy = 100    # minimum cell width (base mesh cell width) in y
dz = 50#100    # minimum cell width (base mesh cell width) in z

x_length = 20000 #34316 #90000.     # domain width in x
y_length = 2000 #34316 #90000.     # domain width in y
z_length = 40000 #16584  #90000.     # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))




# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
M = simpeg.Mesh.TreeMesh([hx, hy, hz], x0='CCC')


# Refine surface topography
[xx, yy] = np.meshgrid(M.vectorNx, M.vectorNy)
[xx, yy,zz] = np.meshgrid([-4000,4000], [-1000,1000],[-100,100])
zz = 0.*(xx**2 + yy**2)  - 0.
#zz = np.zeros([300,300])
pts = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]
M = refine_tree_xyz(
    M, pts, octree_levels=[2, 2], method='surface', finalize=False
    )

# Refine box
xp, yp, zp = np.meshgrid([-600., 600.], [-6500., 6500.], [200., -2600.])
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
#    # The bottom west corner
#    x0 = mesh.x0
#    
#    # The total number of cells
#    nC = mesh.nC
#    
#    # An (nC, 2) array containing the cell-center locations
#    cc = mesh.gridCC
#    
#    # A boolean array specifying which cells lie on the boundary
#    bInd = mesh.cellBoundaryInd
#    
#    # Cell volumes
#    v = mesh.vol   
#    
#    M = mesh  

#print(list(M.vectorCCz))



# DEFININDO MODELO 3D
conds = [2e-1,1e-2]   # [heterogeneidade,background]
sig = simpeg.Utils.ModelBuilder.defineBlock(
    M.gridCC, [-500, -6500, -500], [500, 6500, -2500], conds)
    
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

plt.show()



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
# rx_z = np.array([0.])
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2),  np.zeros((np.prod(rx_x.shape), 1))))

# rx_loc = np.reshape([rx_x,rx_y,rx_z],(1,3))


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
# hx_px_field = h_px_full[0:np.size(M.gridEx,0),:]
# hx_py_field = h_py_full[0:np.size(M.gridEx,0),:]
# hx_px_field = h_px_full[0:np.size(M.gridFx,0),:]
# hx_py_field = h_py_full[0:np.size(M.gridFx,0),:]


# é necessário interpolar o campo h nas arestas

Pbx = M.getInterpolationMat(M.gridEx, 'Fx')
hx_px_field = Pbx*h_px_full
hx_py_field = Pbx*h_py_full



# bb = dict(zip(M.gridEx[:,2],ex_px_field))
# print(bb)

# bb = dict(zip(M.gridEx[:,2],hx_py_field))
# print(bb)





# C = M.edgeCurl
# plt.figure(4)
# plt.spy(C)


ey_px_field = e_px_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
ey_py_field = e_py_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
hy_px_field = h_px_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
hy_py_field = h_py_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]


indx = np.logical_and( abs(M.gridEx[:,2]) < 1e-6, abs(M.gridEx[:,1]) < 1e-6)
indy = np.logical_and( abs(M.gridEy[:,2]) < 1e-4, abs(M.gridEy[:,0]) < 1e-4)



ex_px = ex_px_field[indx]
ex_py = ex_py_field[indx]
hx_px = hx_px_field[indx]
hx_py = hx_py_field[indx]

ey_px = ey_px_field[indy]
ey_py = ey_py_field[indy]
hy_px = hy_px_field[indy]
hy_py = hy_py_field[indy]


#x = M.getTensor('Ex')[0]
x = M.gridEx[indx,0]



ix = 0
Zij = ex_px/hx_py
# Zij = ey_py/hy_px
#Zij = (-ex_px * hx_py + ex_py * hx_px)/(hx_px*hy_py - hx_py*hy_px)
# Zij = ex_px/hy_py
# for i,freq in enumerate(freqs):
    # rho_app[i] = 1/(2*np.pi*freq*mu_0) * abs(Zij[ix,i])**2

# rho_app = 1/(2*np.pi*freqs*mu_0) * abs(Zij[ix,:])**2
# phs     = np.arctan2(Zij[ix,:].imag, Zij[ix,:].real)*(180./np.pi)


rho_app = 1/(2*np.pi*freqs*mu_0) * abs(Zij[:,ix])**2
phs     = np.arctan2(Zij[ix,:].imag, Zij[:,ix].real)*(180./np.pi)



# #plotting
plt.figure()
plt.plot(x,rho_app,xs,rho_ap)
plt.yscale('log')
plt.grid(which = 'minor',axis = 'both')
#plt.ylim((1, 500))
plt.xlim((-3000, 3000))
# plt.plot(x,np.real(hx_py),x,np.imag(hx_py))
# plt.legend(('Real','Imag'))






















