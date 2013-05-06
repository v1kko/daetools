"""********************************************************************************
                            daeDataReporters.py
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
from pyCore import *
from pyDataReporting import *
try:
    import scipy
    from scipy.io import savemat, loadmat
except ImportError, e:
    print 'Cannot load scipy.io.savemat module', str(e)

class daeMatlabMATFileDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectionString, ProcessName):
        self.ProcessName      = ProcessName
        self.ConnectionString = ConnectionString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.Write()
        return True

    def Write(self):
        mdict = {}
        for var in self.Process.Variables:
            mdict[var.Name] = var.Values

        try:
            scipy.io.savemat(self.ConnectionString,
                             mdict,
                             appendmat=False,
                             format='5',
                             long_field_names=False,
                             do_compression=False,
                             oned_as='row')
        except Exception, e:
            print 'Cannot call scipy.io.savemat(); is SciPy installed?\n' + str(e)


class daePlotDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectionString, ProcessName):
        self.ProcessName      = ProcessName
        self.ConnectionString = ConnectionString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        return True

    def Plot(self, *args, **kwargs):
        '''
        args can be either:

            a) instances of daeVariable, or
            b) lists of daeVariable instances, or
            c) a mixture of both.
        Each arg will get its own subplot.  The subplots are all automatically
        arranged such that the resulting figure is as square-like as
        possible.  You can however override the shape by supplying figRows
        and figCols as keyword args.

        Basic Example:

        # Create Log, Solver, DataReporter and Simulation object
        log = daePythonStdOutLog()
        daesolve = daeIDAS()
        from daetools.pyDAE.daeDataReporters import daePlotDataReporter
        datareporter = daePlotDataReporter()
        simulation = simTutorial()

        simulation.m.SetReportingOn(True)
        simulation.ReportingInterval = 20
        simulation.TimeHorizon = 500

        simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        if(datareporter.Connect("", simName) == False):
            sys.exit()

        simulation.Initialize(daesolver, datareporter, log)

        simulation.m.SaveModelReport(simulation.m.Name + ".xml")
        simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

        simulation.SolveInitial()
        simulation.Run()

        simulation.Finalize()
        datareporter.Plot(
            simulation.m.Ci,                       # Subplot 1
            [simulation.m.L, simulation.m.event],  # Subplot 2 (2 sets)
            simulation.m.Vp,                       # Subplot 3
            [simulation.m.L, simulation.m.Vp]      # Subplot 4 (2 sets)
            )
        '''
        try:
            import matplotlib.pyplot as plt
            from math import sqrt, ceil, floor

            ''' Math to make sure we have enough subplots for any number
            of given variables.'''
            n = len(args)

            def kwargCheck(kw='', valueIfNone=0):
                if kwargs.has_key(kw) and kwargs[kw]:
                    return kwargs[kw]
                else:
                    return valueIfNone

            rows = kwargCheck('figRows', valueIfNone=int(round(sqrt(n))))
            cols = kwargCheck('figCols', valueIfNone=int(ceil(sqrt(n))))

            ''' Create lookup for convenience.  There are two kinds of
            "variable" identifiers:
               a) instances of daeDataReceiverVariable (I call processVar)
               b) instances of daeVariable (I call daeVar)
            This lookup will create a name-value dict for later matching.'''
            lookup = {}
            for processVar in self.Process.Variables:
                lookup[processVar.Name] = processVar

            def varLabel(daeVar):
                return '${0} {1}$'.format(daeVar.Name, str(daeVar.VariableType.Units))

            def larger_axlim(axlim):
                """ This function enlarges the y-axis range so that data
                will not lie underneath an axis line.

                    axlim: 2-tuple
                    result: 2-tuple (enlarged range)

                This was taken from (thanks bernie!):
                http://stackoverflow.com/a/6230993/170656
                """
                axmin,axmax = axlim
                axrng = axmax - axmin
                new_min = axmin - 0.1 * axrng
                new_max = axmax + 0.1 * axrng
                return new_min, new_max

            def plotVariable(daeVar):
                ''' Create easy plotter that requires only a daeVariable.
                This function will plot the variable's data onto whatever
                is the current set of axes.  Note that the matchups between
                the processVar and daeVar require daeVar.CanonicalName, not
                merely Name.'''
                if lookup.has_key(daeVar.CanonicalName):
                    processVar = lookup[daeVar.CanonicalName]
                    #import pdb; pdb.set_trace()
                    plt.plot(
                        processVar.TimeValues,
                        processVar.Values,
                        label=varLabel(daeVar))

            for i, arg in enumerate(args):
                '''Loop over arguments: some are variables (each one gets its
                own subplot) and some are lists of variables, these vars
                will share a plot.'''
                axes = plt.subplot(rows, cols, i+1)
                plt.xlabel('Time (s)')
                #plt.title('Title')
                plt.grid(True)
                plt.tight_layout(1.2)
                if isinstance(arg, list):
                    ''' This arg is a list of variables.  These must all be
                    plotted on the same plot.'''
                    variables = arg
                    for v in variables:
                        plotVariable(v)
                    plt.legend()
                else:
                    ''' This arg is a single variable.'''
                    variable = arg
                    plt.ylabel(varLabel(variable))
                    plotVariable(variable)
                axes.set_ylim( larger_axlim( axes.get_ylim() ) )
            plt.show()

        except Exception, e:
            import traceback
            print 'Error: \n' + traceback.format_exc()
