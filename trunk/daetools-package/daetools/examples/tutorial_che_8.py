#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_8.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
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
Model of a gas separation on a porous membrane with a metal support. The model employs
the Generalised Maxwell-Stefan (GMS) equations to predict fluxes and selectivities.
The membrane unit model represents a generic two-dimensonal model of a porous membrane
and consists of four models:

- Retentate compartment (isothermal axially dispersed plug flw)
- Micro-porous membrane
- Macro-porous support layer
- Permeate compartment (the same transport phenomena as in the retentate compartment)

The retentate compartment, the porous membrane, the support layer and the permeate
compartment are coupled via molar flux, temperature, pressure and gas composition at
the interfaces.
The model is described in the section 2.2 Membrane modelling of the following article:

- Nikolic D.D., Kikkinides E.S. (2015) Modelling and optimization of PSA/Membrane
  separation processes. Adsorption 21(4):283-305.
  `doi:10.1007/s10450-015-9670-z <http://doi.org/10.1007/s10450-015-9670-z>`_

and in the original Krishna article:

- Krishna R. (1993) A unified approach to the modeling of intraparticle
  diffusion in adsorption processes. Gas Sep. Purif. 7(2):91-104.
  `doi:10.1016/0950-4214(93)85006-H <http://doi.org/10.1016/0950-4214(93)85006-H>`_

This version is somewhat simplified for it only offers an extended Langmuir isotherm.
The Ideal Adsorption Solution theory (IAS) and the Real Adsorption Solution theory (RAS)
described in the articles are not implemented here.

The problem modelled is separation of hydrocarbons (CH4+C2H6) mixture on a zeolite
(silicalite-1) membrane with a metal support from the section 'Binary mixture permeation'
of the following article:

- van de Graaf J.M., Kapteijn F., Moulijn J.A. (1999) Modeling Permeation of Binary
  Mixtures Through Zeolite Membranes. AIChE J. 45:497–511.
  `doi:10.1002/aic.690450307 <http://doi.org/10.1002/aic.690450307>`_

The CH4 and C2H6 fluxes, and CH4/C2H6 selectivity plots for two cases: GMS and GMS(Dij=∞),
1:1 mixture, and T = 303 K:

.. image:: _static/tutorial_che_8-results.png
   :width: 800px
"""

import os,sys, tempfile, math, numpy, pickle
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *
try:
    from membrane_unit import MembraneUnit
except:
    from .membrane_unit import MembraneUnit

class simTutorial(daeSimulation):
    def __init__(self, Phigh):
        daeSimulation.__init__(self)
        self.m = MembraneUnit("tutorial_che_8")
        self.m.Description = __doc__

        self.Phigh = Phigh

    def SetUpParametersAndDomains(self):
        Nc = 2
        Nz = 20
        Nr = 3

        self.m.Nc.CreateArray(Nc)
        self.m.z.CreateStructuredGrid(Nz, 0.0, 1.0)

        self.m.Tref.SetValue(273.14 * K)
        self.m.Pref.SetValue(1.013E5 * Pa)
        self.m.Tfeed.SetValue(303 * K)

        # Feed compartment
        self.m.F.Nc.CreateArray(Nc)
        self.m.F.z.CreateStructuredGrid(Nz, 0.0, 1.0)

        self.m.F.Rc.SetValue(8.315 * J/(mol*K))
        
        # Membrane
        self.m.M.Nc.CreateArray(Nc)
        self.m.M.z.CreateStructuredGrid(Nz, 0.0, 1.0)
        self.m.M.r.CreateStructuredGrid(Nr, 0.0, 1.0)

        self.m.M.Rc.SetValue(8.315 * J/(mol*K))
        self.m.M.Ro.SetValue(1800 * kg/(m**3))

        self.m.M.Qsat.SetValue(0, (2.456 - 0.002 * self.m.Tfeed.GetValue()) * mol/kg)
        self.m.M.Qsat.SetValue(1, (2.456 - 0.002 * self.m.Tfeed.GetValue()) * mol/kg)

        self.m.M.B.SetValue(0, math.exp(-20.5961) * math.exp(2399.6807 / self.m.Tfeed.GetValue()) * (1/Pa))
        self.m.M.B.SetValue(1, math.exp(-21.7471) * math.exp(3631.2443 / self.m.Tfeed.GetValue()) * (1/Pa))

        # Support
        self.m.S.Nc.CreateArray(Nc)
        self.m.S.z.CreateStructuredGrid(Nz, 0.0, 1.0)
        self.m.S.r.CreateStructuredGrid(Nr, 0.0, 1.0)

        self.m.S.Rc.SetValue(8.315 * J/(mol*K))
        self.m.S.e.SetValue(0.2)

        # Permeate compartment
        self.m.P.Nc.CreateArray(Nc + 1)
        self.m.P.z.CreateStructuredGrid(Nz, 0.0, 1.0)

        self.m.P.Rc.SetValue(8.315 * J/(mol*K))
        
    def SetUpVariables(self):
        Tref   = self.m.Tref.GetQuantity()
        Pref   = self.m.Pref.GetQuantity()
        Tfeed  = self.m.Tfeed.GetQuantity()
        _Tfeed = Tfeed.value
        
        Plow = (1.0e5 * Pa)
        MembraneArea = (2e-4 * (m**2))
        MembraneThickness = (1e-5 * m)
        SupportThickness = (3e-3 * m)
        UnitLength = (0.01 * m)
        Xfeed = [0.50, 0.50]
        Qfeed_stp = (1.667e-6 * (m**3)/s)
        Qsweep_stp = (1.667e-6 * (m**3)/s)

        self.m.MembraneArea.AssignValue(2e-4 * (m**2))
        self.m.MembraneThickness.AssignValue(1e-5 * m)
        self.m.SupportThickness.AssignValue(3e-3 * m)

        self.m.Qfeed_stp.AssignValue(1.667e-6 * (m**3)/s)
        self.m.Qsweep_stp.AssignValue(1.667e-6 * (m**3)/s)
        self.m.Phigh.AssignValue(self.Phigh)
        self.m.Plow.AssignValue(Plow)
        
        self.m.F.Xin.AssignValue(0, 0.50)
        self.m.F.Xin.AssignValue(1, 0.50)
        self.m.F.Length.AssignValue(0.01 * m)
        
        self.m.M.Length.AssignValue(0.01 * m)

        self.m.P.Xin.AssignValue(0, 0.0)
        self.m.P.Xin.AssignValue(1, 0.0)
        self.m.P.Xin.AssignValue(2, 1.0)
        self.m.P.Length.AssignValue(0.01 * m)

        self.m.F.Qin.AssignValue(Qfeed_stp * (Tfeed / Tref) * (Pref / self.Phigh))

        self.m.F.Pin.AssignValue(self.Phigh)
        self.m.F.Area.AssignValue(MembraneArea)

        self.m.M.Area.AssignValue(MembraneArea)
        self.m.M.Thickness.AssignValue(MembraneThickness)

        self.m.S.Thickness.AssignValue(SupportThickness)

        self.m.P.Qin.AssignValue(Qsweep_stp * (Tfeed / Tref) * (Pref / Plow))

        self.m.P.Pin.AssignValue(Plow)
        self.m.P.Area.AssignValue(MembraneArea)
        
        # Feed compartment
        self.m.F.Across.AssignValue(0.004 * (m**2))
        for i in range(0, self.m.F.Nc.NumberOfPoints):
            for z in range(0, self.m.F.z.NumberOfPoints):
                self.m.F.Dz.AssignValue(i, z, 1e-4 * (m**2)/s)

        # Support (start with the Fick law)
        self.m.S.stnOperatingMode.ActiveState = 'sFickLaw'

        self.m.S.Di.AssignValue(0, (1.0e-5 * 0.000304 * (_Tfeed ** 1.75)) * (m**2)/s)
        self.m.S.Di.AssignValue(1, (1.0e-5 * 0.000218 * (_Tfeed ** 1.75)) * (m**2)/s)
        self.m.S.Dij.AssignValue(0, 0, (1.0e-5 * 7.589e-5 * (_Tfeed ** 1.75)) * (m**2)/s)
        self.m.S.Dij.AssignValue(0, 1, (1.0e-5 * 7.589e-5 * (_Tfeed ** 1.75)) * (m**2)/s)
        self.m.S.Dij.AssignValue(1, 0, (1.0e-5 * 7.589e-5 * (_Tfeed ** 1.75)) * (m**2)/s)
        self.m.S.Dij.AssignValue(1, 1, (1.0e-5 * 7.589e-5 * (_Tfeed ** 1.75)) * (m**2)/s)

        # Membrane (start with the Fick law)
        self.m.M.stnOperatingMode.ActiveState = 'sFickLaw'

        self.m.M.Di.AssignValue(0, 1.0e-10 * math.exp(6.0094) * math.exp(-1111.31 / _Tfeed) * (m**2)/s)
        self.m.M.Di.AssignValue(1, 1.0e-10 * math.exp(7.6324) * math.exp(-2190.70 / _Tfeed) * (m**2)/s)

        # Sweep gas flux is zero (no back-flow to the support)
        for z in range(0, self.m.P.z.NumberOfPoints):
            self.m.P.Flux.AssignValue(2, z, 0.0 * mol/((m**2) * s))

        self.m.P.Across.AssignValue(0.004 * (m**2))
        for i in range(0, self.m.P.Nc.NumberOfPoints):
            for z in range(0, self.m.P.z.NumberOfPoints):
                self.m.P.Dz.AssignValue(i, z, 1e-4 * (m**2)/s)

        # Set some important initial guesses
        # These variables affect the results extremely if their initial guesses
        # are very far from the solution 
        self.m.F.Area.SetInitialGuess(2e-4 * (m**2))
        self.m.M.Area.SetInitialGuess(2e-4 * (m**2))
        self.m.P.Area.SetInitialGuess(2e-4 * (m**2))
        self.m.M.Thickness.SetInitialGuess(1e-5 * m)
        self.m.S.Thickness.SetInitialGuess(1e-3 * m)

        self.m.F.Qin.SetInitialGuess(1e-6 * (m**3)/s)
        self.m.P.Qin.SetInitialGuess(1e-6 * (m**3)/s)

    def Run(self):
        # The causes for the model did not work were:
        #  - The Flux equation was not ditributed on eClosedOpen r domain but eOpenClosed by mistake
        #  - cond in Gij was wrong: cond = (i() - k())/(i() - k() + 1E-15)
        #    but should be: cond = 1 - (i() - k())/(i() - k() + 1E-15)
        #  - Too tight tolerances for 'fraction_t' and 'J_theta_t' variable types

        # The model is too non-linear for the solver be able to solve it immeadiately.
        # Therefore we start with the simple case (FickLaw for flux in both support and membrane),
        # solve it, and gradually increase the model complexity towards Maxwell-Stefan equations for both.
        self.m.M.stnOperatingMode.ActiveState = 'sFickLaw'
        self.m.S.stnOperatingMode.ActiveState = 'sFickLaw'
        self.Reinitialize()
        self.Log.Message('Initialised: Membrane(FickLaw) + Support(FickLaw)', 0)
        self.Log.SetProgress(25)

        self.m.M.stnOperatingMode.ActiveState = 'sMaxwellStefan_Dijoo'
        self.m.S.stnOperatingMode.ActiveState = 'sFickLaw'
        self.Reinitialize()
        self.Log.Message('Initialised: Membrane(MaxwellStefan(Dij=∞)) + Support(FickLaw)', 0)
        self.Log.SetProgress(50)

        self.m.M.stnOperatingMode.ActiveState = 'sMaxwellStefan_Dijoo'
        self.m.S.stnOperatingMode.ActiveState = 'sMaxwellStefan'
        self.Reinitialize()
        self.ReportData(1)
        self.Log.Message('Initialised: Membrane(MaxwellStefan(Dij=∞)) + Support(MaxwellStefan) (the results available at t=1s)', 0)
        self.Log.SetProgress(75)

        self.m.M.stnOperatingMode.ActiveState = 'sMaxwellStefan'
        self.m.S.stnOperatingMode.ActiveState = 'sMaxwellStefan'
        self.Reinitialize()
        self.ReportData(2)
        self.Log.Message('Initialised: Membrane(MaxwellStefan) + Support(MaxwellStefan) (the results available at t=2s)', 0)
        self.Log.SetProgress(100)
        self.Log.Message('Done!!', 0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial(3e5 * Pa)
    daesolver = daeIDAS()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 1
    daesolver.RelativeTolerance = 1e-6
    simulator  = daeSimulator(app, simulation=sim, daesolver=daesolver)
    simulator.exec_()

def simulate(Phigh):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(Phigh)
    daesolver    = daeIDAS()

    daesolver.RelativeTolerance = 1e-6

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 1

    dr1 = daeTCPIPDataReporter()
    dr2 = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr1)
    datareporter.AddDataReporter(dr2)

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(dr1.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

    results = dr2.Process.dictVariables
    Flux        = results[simulation.m.Name + '.Membrane.Flux'].Values
    Selectivity = results[simulation.m.Name + '.Selectivity'].Values

    CH4_flux_GMS_Dij_oo             = numpy.average(Flux[1, 0, :])
    C2H6_flux_GMS_Dij_oo            = numpy.average(Flux[1, 1, :])
    CH4_C2H6_selectivity_GMS_Dij_oo = numpy.average(Selectivity[1, 1, 0, :])

    CH4_flux_GMS             = numpy.average(Flux[2, 0, :])
    C2H6_flux_GMS            = numpy.average(Flux[2, 1, :])
    CH4_C2H6_selectivity_GMS = numpy.average(Selectivity[2, 1, 0, :])

    return ((CH4_flux_GMS_Dij_oo, C2H6_flux_GMS_Dij_oo, CH4_C2H6_selectivity_GMS_Dij_oo),
            (CH4_flux_GMS,        C2H6_flux_GMS,        CH4_C2H6_selectivity_GMS))

def full_experiment():
    pressures = []
    fluxes = {'CH4_flux_GMS_Dij_oo':[],'CH4_flux_GMS':[],
              'C2H6_flux_GMS_Dij_oo':[],'C2H6_flux_GMS':[]}
    selectivities = {'CH4_C2H6_GMS_Dij_oo':[],'CH4_C2H6_GMS':[]}
    for Phigh in [1.0e5*Pa, 1.5e5*Pa, 2.0e5*Pa, 2.5e5*Pa, 3.0e5*Pa, 3.5e5*Pa, 4.0e5*Pa, 4.5e5*Pa, 5.0e5*Pa]:
        try:
            results = simulate(Phigh)
            ((CH4_flux_GMS_Dij_oo, C2H6_flux_GMS_Dij_oo, CH4_C2H6_selectivity_GMS_Dij_oo),
             (CH4_flux_GMS,        C2H6_flux_GMS,        CH4_C2H6_selectivity_GMS)) = results

            pressures.append(Phigh.value/1E5)
            fluxes['CH4_flux_GMS_Dij_oo'] .append(CH4_flux_GMS_Dij_oo)
            fluxes['CH4_flux_GMS']        .append(CH4_flux_GMS)
            fluxes['C2H6_flux_GMS_Dij_oo'].append(C2H6_flux_GMS_Dij_oo)
            fluxes['C2H6_flux_GMS']       .append(C2H6_flux_GMS)
            selectivities['CH4_C2H6_GMS_Dij_oo'].append(CH4_C2H6_selectivity_GMS_Dij_oo)
            selectivities['CH4_C2H6_GMS']       .append(CH4_C2H6_selectivity_GMS)

        except Exception as e:
            print(str(e))

    print('Pressures:', pressures)

    pickle.dump(pressures,     open('pressures.pickle', 'wb'))
    pickle.dump(fluxes,        open('fluxes.pickle', 'wb'))
    pickle.dump(selectivities, open('selectivities.pickle', 'wb'))

    pressures     = pickle.load(open('pressures.pickle', 'rb'))
    fluxes        = pickle.load(open('fluxes.pickle', 'rb'))
    selectivities = pickle.load(open('selectivities.pickle', 'rb'))
    #print(pressures)
    #print(fluxes)
    #print(selectivities)

    fontsize = 14
    fontsize_legend = 11
    plt.figure(1, facecolor='white')
    plt.subplot(121)
    plt.plot(pressures, selectivities['CH4_C2H6_GMS_Dij_oo'], 'rs-', label='GMS(Dij=∞)')
    plt.plot(pressures, selectivities['CH4_C2H6_GMS'],        'bo-', label='GMS')
    plt.xlabel('Pressure (kPa)', fontsize=fontsize)
    plt.ylabel('Selectivity CH4/C2H6', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.xlim((1, 6))

    plt.subplot(122)
    plt.plot(pressures, fluxes['CH4_flux_GMS_Dij_oo'],  'rs:', label='CH4-GMS(Dij=∞)')
    plt.plot(pressures, fluxes['CH4_flux_GMS'],         'ro-', label='CH4-GMS')
    plt.plot(pressures, fluxes['C2H6_flux_GMS_Dij_oo'], 'bs:', label='C2H6-GMS(Dij=∞)')
    plt.plot(pressures, fluxes['C2H6_flux_GMS'],        'bo-', label='C2H6-GMS')
    plt.xlabel('Pressure (kPa)', fontsize=fontsize)
    plt.ylabel('Flux mol/(m2.s)', fontsize=fontsize)
    plt.legend(loc=0, fontsize=fontsize_legend)
    plt.xlim((1, 6))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        simulate(3e5 * Pa)
    elif len(sys.argv) > 1 and (sys.argv[1] == 'full_experiment'):
        full_experiment()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
