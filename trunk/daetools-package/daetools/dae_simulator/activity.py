"""********************************************************************************
                            activity.py
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
********************************************************************************"""
import sys, json
from time import localtime, strftime
from daetools.pyDAE import *

class daeActivity(object):    
    @staticmethod
    def simulate(simulation, log                            = None, 
                             datareporter                   = None, 
                             daesolver                      = None, 
                             lasolver                       = None,
                             computeStackEvaluator          = None,
                             timeHorizon                    = 0.0,
                             reportingTimes                 = [],
                             reportingInterval              = 0.0,
                             reportAllVariables             = True,
                             reportTimeDerivatives          = False,
                             reportSensitivities            = False,
                             calculateSensitivities         = False,
                             relativeTolerance              = 0.0,
                             saveModelReport                = False,
                             saveRuntimeModelReport         = False,
                             stopAtModelDiscontinuity       = None,
                             reportDataAroundDiscontinuities= None,
                             lasolver_setoptions_fn         = None,
                             daesolver_setoptions_fn        = None,
                             guiRun                         = False,
                             qtApp                          = None,
                             generate_code_fn               = None,
                             run_after_simulation_init_fn   = None,
                             run_before_simulation_fn       = None,
                             run_after_simulation_fn        = None,
                             initializeAndReturn            = False,
                             **kwargs):
        if reportingInterval <= 0.0:
            raise RuntimeError('Invalid reportingInterval specified: %fs')
        if timeHorizon <= 0.0:
            raise RuntimeError('Invalid timeHorizon specified: %fs')
    
        # Enable reporting of all variables
        simulation.m.SetReportingOn(reportAllVariables)

        # Enable reporting of time derivatives for all reported variables
        simulation.ReportTimeDerivatives = reportTimeDerivatives

        # Enable reporting of sensitivities
        simulation.ReportSensitivities = reportSensitivities
        
        simulation.ReportingInterval = reportingInterval
        simulation.TimeHorizon       = timeHorizon
        if reportingTimes:
            simulation.ReportingTimes = reportingTimes
        
        if stopAtModelDiscontinuity != None:
            simulation.StopAtModelDiscontinuity = stopAtModelDiscontinuity
        
        if reportDataAroundDiscontinuities != None:
            simulation.ReportDataAroundDiscontinuities = reportDataAroundDiscontinuities            

        if computeStackEvaluator:
            simulation.SetComputeStackEvaluator(computeStackEvaluator)
            
        if guiRun:
            if not qtApp:
                qtApp = daeCreateQtApplication(sys.argv)
                
            # Important!
            # Double check sent arguments (whether they are processed at all)
            simulator = daeSimulator(qtApp, simulation                   = simulation,
                                            log                          = log,
                                            datareporter                 = datareporter,
                                            daesolver                    = daesolver,
                                            lasolver                     = lasolver,
                                            relativeTolerance            = relativeTolerance,
                                            calculateSensitivities       = calculateSensitivities,
                                            lasolver_setoptions_fn       = lasolver_setoptions_fn,
                                            daesolver_setoptions_fn      = daesolver_setoptions_fn,
                                            generate_code_fn             = generate_code_fn,
                                            run_after_simulation_init_fn = run_after_simulation_init_fn,
                                            run_before_simulation_fn     = run_before_simulation_fn,
                                            run_after_simulation_fn      = run_after_simulation_fn,
                                            **kwargs)
            simulator.exec_()

        else:
            if not datareporter:
                datareporter = daeTCPIPDataReporter()
                simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if not datareporter.Connect("", simName):
                    raise RuntimeError('Cannot connect TCP/IP data reporter')
            
            if not log:
                log = daePythonStdOutLog()
            
            if not daesolver:
                daesolver = daeIDAS()
            if lasolver:
                if isinstance(lasolver, tuple):
                    if len(lasolver) != 2:
                        raise RuntimeError('Invalid linear solver specified: %s' % lasolver)
                    lasolverType   = lasolver[0]
                    preconditioner = lasolver[1]
                    daesolver.SetLASolver(lasolverType, preconditioner)
                else:
                    daesolver.SetLASolver(lasolver)
                
            if relativeTolerance > 0.0:
                daesolver.RelativeTolerance = relativeTolerance

            # Initialize the simulation
            simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = calculateSensitivities)

            # Save the model report and the runtime model report
            if saveModelReport:
                simulation.m.SaveModelReport(simulation.m.Name + ".xml")
            if saveRuntimeModelReport:
                simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")
        
            if daesolver_setoptions_fn:
                daesolver_setoptions_fn(daesolver)
            
            if lasolver and lasolver_setoptions_fn:
                lasolver_setoptions_fn(lasolver)
            
            if run_after_simulation_init_fn:
                run_after_simulation_init_fn(simulation, log)
            
            if initializeAndReturn:
                #simulation.__aux__objects = [daesolver, log, datareporter, lasolver]
                return simulation
            
            # Solve at time=0 (initialization)
            simulation.SolveInitial()

            # Test OpenCS code generator
            #print('Generating OpenCS model')
            #from daetools.code_generators.opencs import daeCodeGenerator_OpenCS
            #cg = daeCodeGenerator_OpenCS()            
            #options = cg.defaultSimulationOptions_DAE
            #options['LinearSolver']['Preconditioner']['Library'] = 'Ifpack'
            #options['LinearSolver']['Preconditioner']['Name']    = 'Amesos'
            #options['LinearSolver']['Preconditioner']['Parameters'] = {"amesos: solver type": "Amesos_Klu"}
            #cg.generateSimulation(simulation, 
                                  #'OpenCS-' + simulation.m.Name, 
                                  #1,
                                  #simulationOptions = options)
            
            if run_before_simulation_fn:
                run_before_simulation_fn(simulation, log)
            
            if generate_code_fn:
                generate_code_fn(simulation, log)
            
            # Run
            try:
                simulation.Run()
                
            except Exception as e:
                log.Message(str(e), 0)
            
            # Should it be called before or after simulation.Finalize()??
            if run_after_simulation_fn:
                run_after_simulation_fn(simulation, log)
            
            # Finalize
            simulation.Finalize()
            
        return simulation

    @staticmethod
    def optimize(simulation, log                            = None, 
                             datareporter                   = None, 
                             daesolver                      = None, 
                             lasolver                       = None,
                             optimization                   = None,
                             nlpsolver                      = None,
                             computeStackEvaluator          = None,
                             timeHorizon                    = 0.0,
                             reportingInterval              = 0.0,
                             reportingTimes                 = [],
                             reportAllVariables             = True,
                             reportTimeDerivatives          = False,
                             reportSensitivities            = False,
                             stopAtModelDiscontinuity       = None,
                             reportDataAroundDiscontinuities= None,
                             relativeTolerance              = 0.0,
                             saveModelReport                = False,
                             saveRuntimeModelReport         = False,
                             guiRun                         = False,
                             qtApp                          = None,
                             lasolver_setoptions_fn         = None,
                             daesolver_setoptions_fn        = None,
                             nlpsolver_setoptions_fn        = None,
                             initializeAndReturn            = False,
                             **kwargs):
        if reportingInterval <= 0.0:
            raise RuntimeError('Invalid reportingInterval specified: %fs')
        if timeHorizon <= 0.0:
            raise RuntimeError('Invalid timeHorizon specified: %fs')
        if not nlpsolver:
            raise RuntimeError('NLP/MINLP solver must be specified')
    
        # Enable reporting of all variables
        simulation.m.SetReportingOn(reportAllVariables)

        # Enable reporting of time derivatives for all reported variables
        simulation.ReportTimeDerivatives = reportTimeDerivatives

        # Enable reporting of sensitivities
        simulation.ReportSensitivities = reportSensitivities
        
        simulation.ReportingInterval = reportingInterval
        simulation.TimeHorizon       = timeHorizon
        if reportingTimes:
            simulation.ReportingTimes = reportingTimes
        
        if stopAtModelDiscontinuity != None:
            simulation.StopAtModelDiscontinuity = stopAtModelDiscontinuity
        
        if reportDataAroundDiscontinuities != None:
            simulation.ReportDataAroundDiscontinuities = reportDataAroundDiscontinuities            

        if not optimization:
            optimization = daeOptimization()

        if computeStackEvaluator:
            simulation.SetComputeStackEvaluator(computeStackEvaluator)
            
        if guiRun:
            if not qtApp:
                qtApp = daeCreateQtApplication(sys.argv)
                
            # Important!
            # Double check sent arguments (whether they are processed at all)
            simulator = daeSimulator(qtApp, simulation                   = simulation,
                                            optimization                 = optimization,
                                            log                          = log,
                                            datareporter                 = datareporter,
                                            daesolver                    = daesolver,
                                            lasolver                     = lasolver,
                                            nlpsolver                    = nlpsolver,
                                            relativeTolerance            = relativeTolerance,
                                            lasolver_setoptions_fn       = lasolver_setoptions_fn,
                                            daesolver_setoptions_fn      = daesolver_setoptions_fn,
                                            nlpsolver_setoptions_fn      = nlpsolver_setoptions_fn,
                                            **kwargs)
            simulator.exec_()

        else:
            if not datareporter:
                datareporter = daeTCPIPDataReporter()
                simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if not datareporter.Connect("", simName):
                    raise RuntimeError('Cannot connect TCP/IP data reporter')
            
            if not log:
                log = daePythonStdOutLog()
            # Do no print progress
            log.PrintProgress = False
            
            if not daesolver:
                daesolver = daeIDAS()
            if lasolver:
                if isinstance(lasolver, tuple):
                    if len(lasolver) != 2:
                        raise RuntimeError('Invalid linear solver specified: %s' % lasolver)
                    lasolverType   = lasolver[0]
                    preconditioner = lasolver[1]
                    daesolver.SetLASolver(lasolverType, preconditioner)
                else:
                    daesolver.SetLASolver(lasolver)
            
            if relativeTolerance > 0.0:
                daesolver.RelativeTolerance = relativeTolerance
            
            # Initialize the optimization
            optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

            # Save the model report and the runtime model report
            if saveModelReport:
                simulation.m.SaveModelReport(simulation.m.Name + ".xml")
            if saveRuntimeModelReport:
                simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")
        
            if daesolver_setoptions_fn:
                daesolver_setoptions_fn(daesolver)
            
            if lasolver and lasolver_setoptions_fn:
                lasolver_setoptions_fn(lasolver)
            
            if nlpsolver and nlpsolver_setoptions_fn:
                nlpsolver_setoptions_fn(nlpsolver)
            
            if initializeAndReturn:
                return optimization

            # Run
            try:
                optimization.Run()
            except Exception as e:
                log.Message(str(e), 0)
                        
            # Finalize
            optimization.Finalize()
    
        return optimization
