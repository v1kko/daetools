#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_4.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
__doc__ = """
Code verification using the Method of Manufactured Solutions.

Reference: 

1. K. Salari and P. Knupp. Code Verification by the Method of Manufactured Solutions. 
   SAND2000 – 1444 (2000).
   `doi:10.2172/759450 <https://doi.org/10.2172/759450>`_

The problem in this tutorial is the *transient convection-diffusion* equation 
distributed on a rectangular 2D domain with u and v components of velocity:
    
.. code-block:: none

   L(u) = du/dt + (d(uu)/dx + d(uv)/dy) - ni * (d2u/dx2 + d2u/dy2) = 0
   L(v) = dv/dt + (d(vu)/dx + d(vv)/dy) - ni * (d2v/dx2 + d2v/dy2) = 0

The manufactured solutions are: 
    
.. code-block:: none

   um = u0 * (sin(x**2 + y**2 + w0*t) + eps)
   vm = v0 * (cos(x**2 + y**2 + w0*t) + eps)

The terms in the new sources Su and Sv are computed using the daetools derivative 
functions (dt, d and d2).

Again, the Dirichlet boundary conditions are used:
    
.. code-block:: none

   u(LB, y)  = um(LB, y)
   u(UB, y)  = um(UB, y)
   u(x,  LB) = um(x,  LB)
   u(x,  UB) = um(x,  UB)

   v(LB, y)  = vm(LB, y)
   v(UB, y)  = vm(UB, y)
   v(x,  LB) = vm(x,  LB)
   v(x,  UB) = vm(x,  UB)

Steady-state results (w0 = 0.0) for the u-component:
    
.. image:: _static/tutorial_cv_4-u_component.png
   :width: 500px

Steady-state results (w0 = 0.0) for the v-component:
    
.. image:: _static/tutorial_cv_4-v_component.png
   :width: 500px

Numerical vs. manufactured solution plot (u velocity component, 40x32 grid):

.. image:: _static/tutorial_cv_4-results.png
   :width: 500px

The normalised global errors and the order of accuracy plots 
(grids 10x8, 20x16, 40x32, 80x64):

.. image:: _static/tutorial_cv_4-results2.png
   :width: 800px
"""

import sys, numpy
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *
from daetools.solvers.superlu import pySuperLU

no_t = daeVariableType("no_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-6)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")
        self.y  = daeDomain("y", self, m, "Y axis domain")

        self.u0  = 1.0
        self.v0  = 1.0
        self.w0  = 0.0
        self.ni  = 0.7
        self.eps = 0.001

        self.u  = daeVariable("u",  no_t, self, "", [self.x, self.y])
        self.v  = daeVariable("v",  no_t, self, "", [self.x, self.y])
        self.um = daeVariable("um", no_t, self, "", [self.x, self.y])
        self.vm = daeVariable("vm", no_t, self, "", [self.x, self.y])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create some auxiliary functions to make equations more readable 
        u0      = self.u0
        v0      = self.v0
        ni      = self.ni
        w0      = self.w0
        eps     = self.eps
        t       = Time()
        
        u       = lambda x,y: self.u(x,y)
        v       = lambda x,y: self.v(x,y)
        du_dt   = lambda x,y: dt(u(x,y))
        duu_dx  = lambda x,y:  d(u(x,y)*u(x,y), self.x)
        duv_dy  = lambda x,y:  d(u(x,y)*v(x,y), self.y)
        d2u_dx2 = lambda x,y: d2(u(x,y), self.x)
        d2u_dy2 = lambda x,y: d2(u(x,y), self.y)
        
        dv_dt   = lambda x,y: dt(v(x,y))
        dvu_dx  = lambda x,y:  d(v(x,y)*u(x,y), self.x)
        dvv_dy  = lambda x,y:  d(v(x,y)*v(x,y), self.y)
        d2v_dx2 = lambda x,y: d2(v(x,y), self.x)
        d2v_dy2 = lambda x,y: d2(v(x,y), self.y)

        um       = lambda x,y: u0 * (numpy.sin(x()**2 + y()**2 + w0*t) + eps)
        dum_dt   = lambda x,y: dt(um(x,y))
        dumum_dx = lambda x,y:  d(um(x,y)*um(x,y), self.x)
        dumvm_dy = lambda x,y:  d(um(x,y)*vm(x,y), self.y)
        d2um_dx2 = lambda x,y: d2(um(x,y), self.x)
        d2um_dy2 = lambda x,y: d2(um(x,y), self.y)
        
        vm       = lambda x,y: v0 * (numpy.cos(x()**2 + y()**2 + w0*t) + eps)
        dvm_dt   = lambda x,y: dt(vm(x,y))
        dvmum_dx = lambda x,y:  d(vm(x,y)*um(x,y), self.x)
        dvmvm_dy = lambda x,y:  d(vm(x,y)*vm(x,y), self.y)
        d2vm_dx2 = lambda x,y: d2(vm(x,y), self.x)
        d2vm_dy2 = lambda x,y: d2(vm(x,y), self.y)
        
        Su       = lambda x,y: dum_dt(x,y) + (dumum_dx(x,y) + dumvm_dy(x,y)) - ni * (d2um_dx2(x,y) + d2um_dy2(x,y))
        Sv       = lambda x,y: dvm_dt(x,y) + (dvmum_dx(x,y) + dvmvm_dy(x,y)) - ni * (d2vm_dx2(x,y) + d2vm_dy2(x,y))

        # Numerical solution
        eq = self.CreateEquation("u", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = du_dt(x,y) + (duu_dx(x,y) + duv_dy(x,y)) - ni * (d2u_dx2(x,y) + d2u_dy2(x,y)) - Su(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("u(,0)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = u(x,y) - um(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("u(,1)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = u(x,y) - um(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("u(0,)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = u(x,y) - um(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("u(1,)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = u(x,y) - um(x,y)
        eq.CheckUnitsConsistency = False


        # v component
        eq = self.CreateEquation("v", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = dv_dt(x,y) + (dvu_dx(x,y) + dvv_dy(x,y)) - ni * (d2v_dx2(x,y) + d2v_dy2(x,y)) - Sv(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("v(,0)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = v(x,y) - vm(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("v(,1)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = v(x,y) - vm(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("v(0,)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = v(x,y) - vm(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("v(1,)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = v(x,y) - vm(x,y)
        eq.CheckUnitsConsistency = False

        # Manufactured solution
        eq = self.CreateEquation("um", "Manufactured solution")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = self.um(x,y) - um(x,y)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("vm", "Manufactured solution")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = self.vm(x,y) - vm(x,y)
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, Nx, Ny):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_4(%dx%d)" % (Nx,Ny))
        self.m.Description = __doc__
        
        self.Nx = Nx
        self.Ny = Ny

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(self.Nx, -0.1, 0.7)
        self.m.y.CreateStructuredGrid(self.Ny,  0.2, 0.8)
        
    def SetUpVariables(self):
        Nx = self.m.x.NumberOfPoints
        Ny = self.m.y.NumberOfPoints
        
        xp = self.m.x.Points
        yp = self.m.y.Points
        
        u0       = self.m.u0
        v0       = self.m.v0
        eps      = self.m.eps
        
        um0 = lambda x,y: u0 * (numpy.sin(x**2 + y**2) + eps)
        vm0 = lambda x,y: v0 * (numpy.cos(x**2 + y**2) + eps)
        
        for x in range(1, Nx-1):
            for y in range(1, Ny-1):
                self.m.u.SetInitialCondition(x,y, um0(xp[x], yp[y]))
                self.m.v.SetInitialCondition(x,y, vm0(xp[x], yp[y]))
                
# Setup everything manually and run in a console
def simulate(Nx, Ny, **kwargs):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(Nx, Ny)

    # Do no print progress
    log.PrintProgress = False

    lasolver = pySuperLU.daeCreateSuperLUSolver()

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # 1. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    # 2. Data
    dr = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr)

    daeActivity.simulate(simulation, reportingInterval = 2.0, 
                                     timeHorizon       = 90.0,
                                     lasolver          = lasolver,
                                     datareporter      = datareporter,
                                     **kwargs)    
    
    ###########################################
    #  Data                                   #
    ###########################################
    results = dr.Process.dictVariables
    uvar = results[simulation.m.Name + '.u']
    vvar = results[simulation.m.Name + '.v']
    umvar = results[simulation.m.Name + '.um']
    vmvar = results[simulation.m.Name + '.vm']
    times = uvar.TimeValues
    u  = uvar.Values[-1, :, :]  # 3D array [t,x,y]
    v  = vvar.Values[-1, :, :]  # 3D array [t,x,y]
    um = umvar.Values[-1, :, :] # 3D array [t,x,y]
    vm = vmvar.Values[-1, :, :] # 3D array [t,x,y]
    
    return times,u,v,um,vm, simulation

def run(**kwargs):
    Nxs = numpy.array([10, 20, 40, 80])
    Nys = numpy.array([8, 16, 32, 64])
    n = len(Nxs)
    Lx = 0.8
    hs = Lx / Nxs # It's similar in y direction
    Eu = numpy.zeros(n)
    Ev = numpy.zeros(n)
    Cu = numpy.zeros(n)
    Cv = numpy.zeros(n)
    pu = numpy.zeros(n)
    pv = numpy.zeros(n)
    E2 = numpy.zeros(n)
    
    # The normalised global errors
    for i in range(n):
        Nx = int(Nxs[i])
        Ny = int(Nys[i])
        times, u, v, um, vm, simulation = simulate(Nx, Ny, **kwargs)
        Eu[i] = numpy.sqrt((1.0/(Nx*Ny)) * numpy.sum((u-um)**2))
        Ev[i] = numpy.sqrt((1.0/(Nx*Ny)) * numpy.sum((v-vm)**2))

    # Order of accuracy
    for i,Nx in enumerate(Nxs):
        if i == 0:
            pu[i] = 0
            pv[i] = 0
            Cu[i] = 0
            Cv[i] = 0
        else:
            pu[i] = numpy.log(Eu[i]/Eu[i-1]) / numpy.log(hs[i]/hs[i-1])
            pv[i] = numpy.log(Ev[i]/Ev[i-1]) / numpy.log(hs[i]/hs[i-1])
            Cu[i] = Eu[i] / hs[i]**pu[i]
            Cv[i] = Ev[i] / hs[i]**pv[i]
        
    C2u = 0.030 # constant for the second order slope line (to get close to the actual line)
    C2v = 0.075 # constant for the second order slope line (to get close to the actual line)
    Eu2 = C2u * hs**2 # Eu for the second order slope
    Ev2 = C2v * hs**2 # Ev for the second order slope
    
    print('Eu2 =', Eu2)
    print('Ev2 =', Ev2)
    print('pu ', pu)
    print('pv =', pv)
    
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(10,8), facecolor='white')
    grids = ['%dx%d' % (Nx,Ny) for (Nx,Ny) in zip(Nxs,Nys)]
    grids = ','.join(grids)
    fig.canvas.set_window_title('The Normalised global errors and the Orders of accuracy (grids: %s) (cv_4)' % grids)
    
    ax = plt.subplot(221)
    plt.figure(1, facecolor='white')
    plt.loglog(hs, Eu,  'ro', label='Eu(h)')
    plt.loglog(hs, Eu2, 'b-', label='2nd order slope')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('||Eu||', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 0.09))
        
    ax = plt.subplot(222)
    plt.figure(1, facecolor='white')
    plt.semilogx(hs[1:], pu[1:],  'rs-', label='Order of Accuracy (pu)')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('pu', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 0.09))
    plt.ylim((1.94, 2.02))
    
    ax = plt.subplot(223)
    plt.figure(1, facecolor='white')
    plt.loglog(hs, Ev,  'ro', label='Ev(h)')
    plt.loglog(hs, Ev2, 'b-', label='2nd order slope')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('||Ev||', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 0.09))
        
    ax = plt.subplot(224)
    plt.figure(1, facecolor='white')
    plt.semilogx(hs[1:], pv[1:],  'rs-', label='Order of Accuracy (pv)')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('pv', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 0.09))
    plt.ylim((1.94, 2.02))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
