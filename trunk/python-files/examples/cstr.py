#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             cstr.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

"""
"""

import sys
from daetools.pyDAE import *
from daetools.solvers.daeMinpackLeastSq import *
from time import localtime, strftime
from numpy import *
import matplotlib.pyplot as plt

       
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Caf   = daeParameter("Ca_f",  eReal, self, "")
        self.Tc    = daeParameter("T_c",   eReal, self, "")
        self.V     = daeParameter("V",     eReal, self, "")
        self.ro    = daeParameter("&rho;", eReal, self, "")
        self.cp    = daeParameter("cp",    eReal, self, "")
        self.R     = daeParameter("R",     eReal, self, "")
        self.UA    = daeParameter("UA",    eReal, self, "")

        # Parameters to estimate
        self.dH    = daeVariable("&Delta;H_r", no_t, self, "")
        self.E     = daeVariable("E",          no_t, self, "")
        self.k0    = daeVariable("k_0",        no_t, self, "")
        
        # Inputs
        self.q     = daeVariable("q",    molar_flowrate_t,      self, "")
        self.Tf    = daeVariable("T_f",  temperature_t,         self, "")
        
        # Measured variables
        self.T  = daeVariable("T",  temperature_t,         self, "")
        self.Ca = daeVariable("Ca", molar_concentration_t, self, "")

    def DeclareEquations(self):
        eq = self.CreateEquation("MolarBalance", "")
        eq.Residual = self.V() * self.Ca.dt() - \
                      self.q() * (self.Caf() - self.Ca()) + \
                      1E10*self.k0() * self.V() * Exp(-(1E4*self.E()) / (self.R() * self.T())) * self.Ca()

        eq = self.CreateEquation("HeatBalance", "")
        eq.Residual = self.ro() * self.cp() * self.V() * self.T.dt() - \
                      self.q()  * self.ro() * self.cp() * (self.Tf() - self.T()) - \
                      self.V()  * 1E4 * self.dH() * 1E10*self.k0() * Exp(-(1E4*self.E())/(self.R() * self.T())) * self.Ca() - \
                      self.UA() * (self.Tc() - self.T())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("cstr")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.Caf.SetValue(1)
        self.m.Tc.SetValue(270)
        self.m.ro.SetValue(1000)
        self.m.cp.SetValue(0.239)
        self.m.V.SetValue(100)
        self.m.R.SetValue(8.31451)
        self.m.UA.SetValue(5e4)

    def SetUpVariables(self):
        # Inputs
        self.m.q.AssignValue(100)
        self.m.Tf.AssignValue(350)
        
        # Parameters 
        self.m.dH.AssignValue(5)         # * 1E4
        self.m.E.AssignValue(7.27519625) # * 1E4
        self.m.k0.AssignValue(7.2)       # * 1E10

        self.m.T.SetInitialCondition(305)
        self.m.Ca.SetInitialCondition(0.9)
    
    def SetUpParameterEstimation(self):
        self.SetMeasuredVariable(self.m.Ca)
        self.SetMeasuredVariable(self.m.T)
        
        self.SetInputVariable(self.m.Tf)
        self.SetInputVariable(self.m.q)
        
        self.SetModelParameter(self.m.dH, -10, 10,  3.0)
        self.SetModelParameter(self.m.E,  -10, 10, 10.0)
        self.SetModelParameter(self.m.k0, -10, 10, 10.0)

if __name__ == "__main__":
    try:
        log          = daePythonStdOutLog()
        daesolver    = daeIDAS()
        datareporter = daeTCPIPDataReporter()
        simulation   = simTutorial()
        minpack      = daeMinpackLeastSq()

        # Enable reporting of all variables
        simulation.m.SetReportingOn(True)

        # Set the time horizon and the reporting interval
        simulation.ReportingInterval = 2
        simulation.TimeHorizon = 10

        # Connect data reporter
        simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        if(datareporter.Connect("", simName) == False):
            sys.exit()

        # Some info about the experiments
        Nparameters          = 3
        Ninput_variables     = 2
        Nmeasured_variables  = 2
        Nexperiments         = 9
        Ntime_points         = 5

        # Experimental data:
        data = [
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [300.0, 80.0],
                    [[0.97732195658793863, 0.99382761129635511, 0.9971545852414353, 0.9978200659445241, 0.99795141821925282], 
                    [278.50287635180217, 278.41462045030585, 278.41505803940998, 278.41522503544002, 278.41525337694367]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [350.0, 80.0],
                    [[0.97154663104358974, 0.98693919781932038, 0.99000039991628508, 0.99060677344976489, 0.99072791710437746], 
                    [292.70308620725126, 292.66152000549755, 292.66388868120322, 292.66438921651689, 292.66446031579858]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [400.0, 80.0],
                    [[0.94926988426299097, 0.95733743128649029, 0.95881631119436961, 0.95908688799462405, 0.9591397959777177], 
                    [308.24333375234033, 308.31506420994157, 308.32308886188093, 308.3244588191435, 308.32472199190562]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [300.0, 100.0],
                    [[0.98423763220581939, 0.99624175916252067, 0.99785934013575761, 0.99807447521841319, 0.99810543937631713], 
                    [279.88582113257195, 279.82980257703701, 279.83015616568571, 279.83020653270233, 279.83020764015703]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [350.0, 100.0],
                    [[0.97653638028997192, 0.98736325854701645, 0.98879332575588563, 0.98897709690476976, 0.98900197570821757],
                    [296.63070059597339, 296.61448339381525, 296.61629972946668, 296.61654843841842, 296.61657954901625]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [400.0, 100.0],
                    [[0.93540400392674161, 0.93389769972253289, 0.93360470373895299, 0.93357528353909969, 0.93357220324321666], 
                    [316.33284141035926, 316.53808152856362, 316.53861078553973, 316.53781394674996, 316.53774998619286]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [300.0, 120.0],
                    [[0.98888455707714884, 0.99735652730280688, 0.99811830630270848, 0.99818614563997532, 0.9981925622615182], 
                    [281.10836086186231, 281.07311420606504, 281.07337457397858, 281.07341709279234, 281.07340242412027]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [350.0, 120.0],
                    [[0.97907836158806061, 0.98636371669339051, 0.9870069933513016, 0.98705719777795065, 0.9870638393430804], 
                    [300.1450376003076, 300.14640429635767, 300.14754885395024, 300.14763681376814, 300.14764772547164]]
                ), 
                (
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [400.0, 120.0],
                    [[0.9004152748567531, 0.87761038942822822, 0.87535452144816661, 0.87547304059962017, 0.87551686721353328], 
                    [325.86070801501398, 326.93848460304991, 326.90042861427992, 326.88096791516557, 326.8794366002345]]
                )
            ]
       
        # Initialize MinpackLeastSq
        minpack.Initialize(simulation, 
                           daesolver, 
                           datareporter, 
                           log, 
                           PrintResidualsAndJacobian = False,
                           ftol                      = 1E-8,
                           xtol                      = 1E-8,
                           factor                    = 0.5,
                           experimental_data         = data)
        
        # Save the model report and the runtime model report
        simulation.m.SaveModelReport(simulation.m.Name + ".xml")
        simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

        # Run
        minpack.Run()

        # Print the results
        print 'Status:', minpack.msg
        print 'Number of function evaluations =', minpack.infodict['nfev']
        print 'Root mean square deviation =', minpack.rmse
        print 'Estimated parameters values:', minpack.p_estimated

        # Plot the comparison between the measured and fitted data
        nExp = 5 # plot the 6th experiment
        values = minpack.infodict['fvec'].reshape( (Nmeasured_variables, Nexperiments, Ntime_points) )
        
        x_axis = data[nExp][0]
        Ca_fit = data[nExp][2][0] + values[0, nExp, :]
        T_fit  = data[nExp][2][1] + values[1, nExp, :]
        Ca_exp = data[nExp][2][0]
        T_exp  = data[nExp][2][1]

        fig = plt.figure()
        
        yprops  = dict()
        axprops = dict()

        ax1 = fig.add_axes([0.10, 0.55, 0.85, 0.40], **axprops)
        ax1.plot(x_axis, Ca_fit, 'blue', x_axis, Ca_exp, 'o')
        ax1.set_ylabel('Ca', **yprops)
        ax1.legend(['Ca-fit', 'Ca-exp'], frameon=False)

        axprops['sharex'] = ax1
        #axprops['sharey'] = ax1
        
        ax2 = fig.add_axes([0.10, 0.05, 0.85, 0.40], **axprops)
        ax2.plot(x_axis, T_fit, 'green', x_axis, T_exp, 'o')
        ax2.set_ylabel('T', **yprops)
        ax2.legend(['T-fit', 'T-exp'], frameon=False)
    
        plt.show()
    
    except Exception, e:
        print str(e)
        
    finally:
        minpack.Finalize()
