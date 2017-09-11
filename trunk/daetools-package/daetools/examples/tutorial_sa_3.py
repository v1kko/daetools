#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_sa_3.py
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
__doc__ = """
This tutorial illustrates the global variance-based sensitivity analysis methods
available in the `SALib <https://github.com/SALib/SALib>`_ python library.

The problem is adopted from the section 2.6 of the following article:
    
- A. Saltelli, M. Ratto, S. Tarantola, F. Campolongo.
  Sensitivity Analysis for Chemical Models. Chem. Rev. (2005), 105(7):2811-2828.
  `doi:10.1021/cr040659d <http://dx.doi.org/10.1021/cr040659d>`_
  
The model describes a thermal analysis of a batch reactor, with exothermic 
reaction A -> B. The model equations are written in dimensionless form.

Three global sensitivity analysis methods available in SALib are applied:
    
- Morris (Elementary Effect/Screening method)
- Sobol (Variance-based methods)
- FAST (Variance-based methods)

Results from the sensitivity analysis:
    
.. code-block:: none  

 -------------------------------------------------------
        Morris (N = 510)
 -------------------------------------------------------
     Param          mu        mu*   mu*_conf      Sigma
         B    0.367412   0.367412   0.114161   0.546276
     gamma   -0.040556   0.056616   0.021330   0.111878
       psi    0.311563   0.311563   0.103515   0.504398
   theta_a    0.326932   0.326932   0.102303   0.490423
   theta_0    0.021208   0.023524   0.016015   0.074062
 -------------------------------------------------------

 -------------------------------------------------------
        Sobol (N = 6144)
 -------------------------------------------------------
     Param          S1    S1_conf         ST    ST_conf
         B    0.094110   0.078918   0.581930   0.154737
     gamma   -0.002416   0.012178   0.044354   0.027352
       psi    0.171043   0.087782   0.524579   0.142115
   theta_a    0.072535   0.042848   0.523394   0.165736
   theta_0    0.002340   0.004848   0.008173   0.005956

 Parameter pairs          S2    S2_conf
         B/gamma    0.180427   0.160979
           B/psi    0.260689   0.171367
       B/theta_a    0.143261   0.154060
       B/theta_0    0.177129   0.156582
       gamma/psi    0.000981   0.027443
   gamma/theta_a    0.004956   0.036554
   gamma/theta_0   -0.009392   0.027390
     psi/theta_a    0.166155   0.172727
     psi/theta_0   -0.016434   0.129177
 theta_a/theta_0    0.109057   0.127287
 -------------------------------------------------------

 ---------------------------------
        FAST (N = 6150)
 ---------------------------------
     Param          S1         ST
         B    0.135984   0.559389
     gamma    0.000429   0.029026
       psi    0.171291   0.602461
   theta_a    0.144617   0.534116
   theta_0    0.000248   0.040741
 ---------------------------------


The scatter plot for the Morris method:
    
.. image:: _static/tutorial_sa_3-scatter_morris.png
   :width: 800px

The scatter plot for the Sobol method:
    
.. image:: _static/tutorial_sa_3-scatter_sobol.png
   :width: 800px

The scatter plot for the FAST method:
    
.. image:: _static/tutorial_sa_3-scatter_fast.png
   :width: 800px
"""

import os, sys, numpy, time
from daetools.pyDAE import *
from time import localtime, strftime
# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l
import matplotlib.pyplot as plt

k_t                  = daeVariableType("k_t",                    s**(-1),  0.0, 1E20, 0, 1e-05)
mass_concentration_t = daeVariableType("mass_concentration_t", kg/(m**3),  0.0, 1E20, 0, 1e-05)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.n       = daeParameter("n",    unit(), self, "")
        
        self.B       = daeVariable("B",       no_t, self, "")
        self.theta_a = daeVariable("theta_a", no_t, self, "")
        self.psi     = daeVariable("psi",     no_t, self, "")
        self.gamma   = daeVariable("gamma",   no_t, self, "")
        self.x       = daeVariable("x",       no_t, self, "")
        self.theta   = daeVariable("theta",   no_t, self, "")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        B         = self.B()
        theta     = self.theta()
        theta_a   = self.theta_a()
        gamma     = self.gamma()
        x         = self.x()
        psi       = self.psi()
        n         = self.n()
        t         = Time()
        dx_dt     = dt(x)
        dtheta_dt = dt(theta)
        
        # Reaction rate constants
        eq = self.CreateEquation("x", "")
        eq.Residual = dx_dt - Exp(theta / (1 + theta/gamma)) * ((1 - x) ** n)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("theta", "")
        eq.Residual = dtheta_dt - ( B * Exp(theta / (1 + theta/gamma)) * ((1 - x) ** n) - B/psi * (theta - theta_a) )
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, n, B, gamma, psi, theta_a, x_0, theta_0):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_sa_3")
        self.m.Description = __doc__
        
        self.n       = n
        self.B       = B
        self.gamma   = gamma
        self.psi     = psi
        self.theta_a = theta_a
        self.x_0     = x_0
        self.theta_0 = theta_0

    def SetUpParametersAndDomains(self):
        self.m.n.SetValue(self.n)
    
    def SetUpVariables(self):
        self.m.B.AssignValue(self.B)
        self.m.gamma.AssignValue(self.gamma)
        self.m.psi.AssignValue(self.psi)
        self.m.theta_a.AssignValue(self.theta_a)
        
        self.m.x.SetInitialCondition(self.x_0)
        self.m.theta.SetInitialCondition(self.theta_0)

def simulate_p(args):
    run_no, n, B, gamma, psi, theta_a, x_0, theta_0, sequential = args
    return simulate(run_no, n, B, gamma, psi, theta_a, x_0, theta_0, sequential)

def simulate(run_no, n, B, gamma, psi, theta_a, x_0, theta_0, sequential):
    #print('run_no %d on tid=%s' % (run_no, os.getpid()))
    
    # If running in parallel disable OpenMP for evaluation of residuals and derivatives!
    # Running simulations in multiple OpenMP threads where all of them 
    # use OpenMP at the same time for calculations actually decreases 
    # the overall performance.
    cfg  = daeGetConfig()
    if not sequential:
        cfg.SetBoolean("daetools.core.equations.parallelEvaluation", False)
        
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(n, B, gamma, psi, theta_a, x_0, theta_0)

    # Do no print progress
    log.PrintProgress = False
    log.Enabled = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True
    
    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.002 # 
    simulation.TimeHorizon       = 0.3   # 0.5 hour

    # Connect data reporter
    simName = '%s [%d]-%s' % (simulation.m.Name, run_no, strftime("[%d.%m.%Y %H:%M:%S]", localtime()))

    # 1. Disable TCP/IP to improve the performance!
    #tcpipDataReporter = daeTCPIPDataReporter()
    #datareporter.AddDataReporter(tcpipDataReporter)
    #if not tcpipDataReporter.Connect("", simName):
    #    sys.exit()

    # 2. Data
    dr = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr)

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()
    
    ###########################################
    #  Data                                   #
    ###########################################
    # Get a dictionary with the reported variables
    variables = dr.Process.dictVariables
    
    # Auxiliary functions
    def variable(variableName):
        return variables['tutorial_sa_3.%s' % variableName]

    theta_var = variable('theta')
    theta = theta_var.Values[:]
    
    return numpy.max(theta)

def run(**kwargs):
    ###################################################################
    # 1. Selection of the global sensitivity method.
    #    Available methods:
    #     - Morris (Elementary Effect/Screening method)
    #     - Sobol  (Variance based method)
    #     - FAST   (Variance based method)
    ###################################################################
    SA_method = 'Sobol'
    
    ###################################################################
    # 2. Definition of the model inputs (used by all methods).
    ###################################################################
    problem = {
        'num_vars': 5,
        'names': ['B', 'gamma', 'psi', 'theta_a', 'theta_0'],
        'bounds': [[10, 30],    # B
                   [15, 25],    # gamma
                   [0.4, 0.6],  # psi
                   [-0.2, 0.2], # theta_a
                   [-0.2, 0.2]  # theta_0
                  ]
              }
    
    ###################################################################
    # 3. Generation of samples (Morris, FAST or Sobol sampling).
    ###################################################################
    if SA_method == 'Morris':
        from daetools.ext_libs.SALib.sample.morris  import sample
        from daetools.ext_libs.SALib.analyze.morris import analyze
        
        # Generate samples using the Morris method.
        # Sample size n=85, no. params k=5 => N=(k+1)*n=510 (512 in the referenced article).
        param_values = sample(problem, 85, num_levels=4, grid_jump=2)
        
        N = param_values.shape[0]

        B       = param_values[:,0] 
        gamma   = param_values[:,1] 
        psi     = param_values[:,2]
        theta_a = param_values[:,3] 
        theta_0 = param_values[:,4]
    
    elif SA_method == 'Sobol':
        from daetools.ext_libs.SALib.sample.saltelli import sample
        from daetools.ext_libs.SALib.analyze.sobol   import analyze
        
        # Generate samples using a Saltelli's extension of the Sobol sequence.
        # Sample size n=512, no. params k=5 => N=(2k+2)*n=6144 (as in the referenced article).
        param_values = sample(problem, 512, calc_second_order=True)
        
        N = param_values.shape[0]
        
        B       = param_values[:,0] 
        gamma   = param_values[:,1] 
        psi     = param_values[:,2]
        theta_a = param_values[:,3] 
        theta_0 = param_values[:,4]
    
    elif SA_method == 'FAST':
        from daetools.ext_libs.SALib.sample.fast_sampler import sample
        from daetools.ext_libs.SALib.analyze.fast        import analyze
        
        # Generate samples using a FAST method.
        # Sample size n=1230, no. params k=5 => N=k*n=6150.
        param_values = sample(problem, 1230, M=4)
        
        N = param_values.shape[0]
        
        B       = param_values[:,0] 
        gamma   = param_values[:,1] 
        psi     = param_values[:,2]
        theta_a = param_values[:,3] 
        theta_0 = param_values[:,4]
           
    else: 
        # Generate sample manually using numpy.random functions.
        # Only scatter plots will be generated (no Sensitivity Analysis).
        N = 6144
        
        B       = numpy.random.normal (loc=20.0, scale=4.0, size=N)
        gamma   = numpy.random.normal (loc=20.0, scale=2.0, size=N)
        psi     = numpy.random.uniform(low=0.4,  high=0.6,  size=N)
        theta_a = numpy.random.normal (loc=0.0,  scale=0.2, size=N)
        theta_0 = numpy.random.normal (loc=0.0,  scale=0.2, size=N)

    #print(N)

    ###################################################################
    # 4. Generation of outputs for a given input matrix (daetools).
    ###################################################################
    theta_max = numpy.zeros(N)
        
    def evaluate_sequential():
        print('Running %d simulations sequentially...' % N)
        for i in range(N): 
            theta_max[i] = simulate(run_no  = i,
                                    n       = 1, 
                                    B       = B[i], 
                                    gamma   = gamma[i], 
                                    psi     = psi[i], 
                                    theta_a = theta_a[i], 
                                    x_0     = 0.0, 
                                    theta_0 = theta_0[i],
                                    sequential = True)
            print('  Finished simulation run %d' % i)
    
    def evaluate_parallel():
        from multiprocessing import Pool
        # Create a pool of workers to calculate N outputs
        # Don't forget to disable OpenMP for evaluation of residuals and derivatives!!
        pool = Pool()
        args = [(i, 1, B[i], gamma[i], psi[i], theta_a[i], 0.0, theta_0[i], False) for i in range(N)]
        print('Running %d simulations in parallel...' % N)
        theta_max[:] = pool.map(simulate_p, args, chunksize=1)
    
    start = time.time()
    evaluate_parallel()
    end = time.time()
    print('Time to evaluate %d simulations: %fs' % (N, end-start))
    
    # Transform theta_max: (T0-T)*gamma/T0 into the Max. rise in temperature: (Tmax-T0)/T0
    max_rise_T = theta_max / gamma
    
    ###################################################################
    # 5. Performing Sensitivity Analysis (Morris, FAST or Sobol).
    ###################################################################
    num_vars = problem['num_vars']
    names = problem['names']
    
    print('')
    if SA_method == 'Morris':
        res = analyze(problem, param_values, max_rise_T, conf_level=0.95, print_to_console=True, num_levels=4, grid_jump=2)
        
        mu           = res['mu']
        mu_star      = res['mu_star']
        mu_star_conf = res['mu_star_conf']
        S            = res['sigma']
        
        print('-------------------------------------------------------')
        print('       %s (N = %d)' % (SA_method, N))
        print('-------------------------------------------------------')
        print('%10s  %10s %10s %10s %10s' % ('Param', 'mu', 'mu*', 'mu*_conf', 'Sigma'))
        for i in range(num_vars):
            print('%10s  %10f %10f %10f %10f' % (names[i], mu[i], mu_star[i], mu_star_conf[i], S[i]))
        print('-------------------------------------------------------')
    
    elif SA_method == 'Sobol':
        res = analyze(problem, max_rise_T, print_to_console=False)
        
        S1      = res['S1']
        S1_conf = res['S1_conf']
        ST      = res['ST']
        ST_conf = res['ST_conf']
        S2      = res['S2']
        S2_conf = res['S2_conf']
        
        print('-------------------------------------------------------')
        print('       %s (N = %d)' % (SA_method, N))
        print('-------------------------------------------------------')
        print('%10s  %10s %10s %10s %10s' % ('Param', 'S1', 'S1_conf', 'ST', 'ST_conf'))
        for i in range(num_vars):
            print('%10s  %10f %10f %10f %10f' % (names[i], S1[i], S1_conf[i], ST[i], ST_conf[i]))

        print('')
        print('%16s  %10s %10s' % ('Parameter pairs', 'S2', 'S2_conf'))
        # S2 matrix is symmetric so take care of its symmetry while printing interactions
        for i in range(num_vars):
            for j in range(i+1, num_vars):
                print('%16s  %10f %10f' % (names[i]+'/'+names[j], S2[i,j], S2_conf[i,j]))
        print('-------------------------------------------------------')
    
    elif SA_method == 'FAST':
        res = analyze(problem, max_rise_T, M=4, print_to_console=True)
        
        S1 = res['S1']
        ST = res['ST']
        
        print('---------------------------------')
        print('       %s (N = %d)' % (SA_method, N))
        print('---------------------------------')
        print('%10s  %10s %10s' % ('Param', 'S1', 'ST'))
        for i in range(num_vars):
            print('%10s  %10f %10f' % (names[i], S1[i], ST[i]))
        print('---------------------------------')
    
    ###################################################################
    # 6. Generation of scatter plots.
    ###################################################################
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(10,8), facecolor='white')

    ax = plt.subplot(231)
    plt.scatter(B, max_rise_T, c='b', s=6)
    plt.xlabel('B', fontsize=fontsize)
    plt.ylabel('max rise in T', fontsize=fontsize)
    plt.xlim((10, 30))
    
    ax = plt.subplot(232)
    plt.scatter(gamma, max_rise_T, c='r', s=6)
    plt.xlabel('gamma', fontsize=fontsize)
    plt.ylabel('max rise in T', fontsize=fontsize)
    plt.xlim((15, 25))
    
    ax = plt.subplot(233)
    plt.scatter(theta_a, max_rise_T, c='g', s=6)
    plt.xlabel('theta_a', fontsize=fontsize)
    plt.ylabel('max rise in T', fontsize=fontsize)
    plt.xlim((-0.2, 0.2))
    
    ax = plt.subplot(234)
    plt.scatter(theta_0, max_rise_T, c='k', s=6)
    plt.xlabel('theta_0', fontsize=fontsize)
    plt.ylabel('max rise in T', fontsize=fontsize)
    plt.xlim((-0.2, 0.2))
    
    ax = plt.subplot(235)
    plt.scatter(psi, max_rise_T, c='m', s=6)
    plt.xlabel('psi', fontsize=fontsize)
    plt.ylabel('max rise in T', fontsize=fontsize)
    plt.xlim((0.4, 0.6))
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    run()
