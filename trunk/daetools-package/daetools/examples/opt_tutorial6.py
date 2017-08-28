#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            opt_tutorial6.py
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
daeMinpackLeastSq module test.
"""

import sys, numpy
from time import localtime, strftime
from numpy.linalg import cholesky
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from daetools.pyDAE import *
from daetools.solvers.minpack import daeMinpackLeastSq

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Model inputs
        self.x     = daeVariable("x", no_t, self)
        
        # Model outputs (measured quantities)
        self.y     = daeVariable("y", no_t, self)
        
        # Model parameters
        self.A     = daeVariable("A", no_t, self)
        self.k     = daeVariable("k", no_t, self)
        self.theta = daeVariable("&theta;", no_t, self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("y")
        eq.Residual = self.y() - self.A() * Sin(2 * numpy.pi * self.k() * self.x() + self.theta())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial6")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x.AssignValue(1)
        self.m.A.AssignValue(1)
        self.m.k.AssignValue(1)
        self.m.theta.AssignValue(1)

    def SetUpParameterEstimation(self):
        self.SetMeasuredVariable(self.m.y)
        
        self.SetInputVariable(self.m.x)
        
        self.SetModelParameter(self.m.A,      1.0,  50.0, 10.0)
        self.SetModelParameter(self.m.k,     10.0, 100.0, 33.0)
        self.SetModelParameter(self.m.theta,  0.1,   2.0,  0.5)

def plotConfidenceEllipsoid(minpack, x_param_index, y_param_index, confidence, x_label, y_label):
    fig = plt.figure()
    x_ellipse, y_ellipse, x0, y0 = minpack.getConfidenceEllipsoid(x_param_index = x_param_index, 
                                                                  y_param_index = y_param_index, 
                                                                  confidence    = confidence)
    ax = fig.add_subplot(111, aspect='auto')
    ax.plot(x_ellipse, y_ellipse)
    ax.plot(x0, y0, 'o')

    legend = [str(confidence)+'%', 'opt']
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.legend(legend) #, frameon=False)
    return fig

def plotExpFitComparison(minpack, input_variable_index, measured_variable_index, x_label, y_label, legend):
    fig = plt.figure()
    x_axis, y_exp, y_fit = minpack.getFit_SS(input_variable_index = input_variable_index, measured_variable_index = measured_variable_index)
    ax = fig.add_subplot(111, aspect='auto')
    ax.plot(x_axis, y_fit, 'blue', x_axis, y_exp, 'o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(legend) #, frameon=False)
    return fig
    
def run(guiRun = False, qtApp = None):
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    minpack      = daeMinpackLeastSq()

    # Do no print progress
    log.PrintProgress = False
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.5
    simulation.TimeHorizon = 10

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Some info about the experiments
    Ntime_points         = 1
    Nparameters          = 3
    Ninput_variables     = 1
    Nmeasured_variables  = 1
    Nexperiments         = 30

    # Experimental data template
    # Tuple ([time intervals], [input values], [measured values]):
    #   ( 
    #     [t_0, t_1, ..., t_tn], [x_0, x_1, ..., x_i], [ [y_00, y_01, ..., y_0tn],
    #                                                    [y_10, y_11, ..., y_1tn],
    #
    #                                                    [y_m0, y_m1, ..., y_mtn]
    #                                                  ]
    #   )
    #   where indexes are:
    #     tn = Ntime_intervals
    #     m  = Nmeasured_variables
    #     i  = Ninput_variables
    data = [
            (
                [1.0], [0.000], [[5.9567423599999998]]
            ), 
            (
                [1.0], [0.002], [[10.03610565]]
            ), 
            (
                [1.0], [0.004], [[10.14475642]]
            ),
            (
                [1.0], [0.006], [[9.1672252099999998]]
            ),
            (
                [1.0], [0.008], [[8.5209392899999994]]
            ), 
            (
                [1.0], [0.010], [[4.7884286300000003]]
            ), 
            (
                [1.0], [0.012], [[2.8746775499999999]]
            ), 
            (
                [1.0], [0.014], [[-3.9342732499999999]]
            ), 
            (
                [1.0], [0.016], [[-6.1307100999999999]]
            ), 
            (
                [1.0], [0.018], [[-9.2616808299999995]]
            ), 
            (
                [1.0], [0.020], [[-9.2527247500000005]]
            ), 
            (
                [1.0], [0.022], [[-10.428504139999999]]
            ), 
            (
                [1.0], [0.024], [[-4.7117558700000002]]
            ), 
            (
                [1.0], [0.026], [[-3.6040301299999999]]
            ), 
            (
                [1.0], [0.028], [[-0.1103975]]
            ), 
            (
                [1.0], [0.030], [[3.8037288999999999]]
            ),
            (
                [1.0], [0.032], [[8.5151208199999999]]
            ), 
            (
                [1.0], [0.034], [[9.7823271799999993]]
            ), 
            (
                [1.0], [0.036], [[9.9193174699999993]]
            ), 
            (
                [1.0], [0.038], [[5.1710806099999997]]
            ), 
            (
                [1.0], [0.040], [[6.4746835999999997]]
            ), 
            (
                [1.0], [0.042], [[0.66528089000000001]]
            ), 
            (
                [1.0], [0.044], [[-5.1034402700000001]]
            ), 
            (
                [1.0], [0.046], [[-7.12668123]]
            ), 
            (
                [1.0], [0.048], [[-9.4208056599999992]]
            ), 
            (
                [1.0], [0.050], [[-8.2317054299999999]]
            ), 
            (
                [1.0], [0.052], [[-6.5608158999999997]]
            ), 
            (
                [1.0], [0.054], [[-6.28524014]]
            ), 
            (
                [1.0], [0.056], [[-2.3024634000000002]]
            ), 
            (
                [1.0], [0.058], [[-0.79571451999999998]]
            )
            ]       
    
    # Initialize MinpackLeastSq
    minpack.Initialize(simulation, 
                       daesolver,
                       datareporter,
                       log,
                       experimental_data            = data,
                       print_residuals_and_jacobian = True,
                       enforce_parameters_bounds    = False,
                       minpack_leastsq_arguments    = {'ftol'   : 1E-8,
                                                       'xtol'   : 1E-8,
                                                       'factor' : 100.0} )
    
    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Run
    minpack.Run()

    # Print the results
    print('Status:', minpack.msg)
    print('Number of function evaluations =', minpack.infodict['nfev'])
    print('Estimated parameters\' values:', minpack.p_estimated)
    print('The real parameters\' values:', ['10.0', '33.33333333', '0.523598333'])
    print('chi2 =', minpack.x2)
    print('Standard deviation =', minpack.sigma)
    print('Covariance matrix:')
    print(minpack.cov_x)
    
    # Plot 90, 95, and 99% confidence ellipsoids
    plotConfidenceEllipsoid(minpack, 0, 1, 95, 'A', 'k')
    plotConfidenceEllipsoid(minpack, 0, 2, 95, 'A', 'theta')
    plotConfidenceEllipsoid(minpack, 1, 2, 95, 'k', 'theta')
    
    # Plot exp-fit comparison for y = f(x)
    plotExpFitComparison(minpack, 0, 0, 'x', 'y', ['y-fit', 'y-exp'])
    
    plt.show()
        
    minpack.Finalize()

if __name__ == "__main__":
    run()
