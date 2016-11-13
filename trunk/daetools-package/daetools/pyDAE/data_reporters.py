"""********************************************************************************
                            data_reporters.py
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
********************************************************************************"""
import os, sys, numpy
from pyCore import *
from pyDataReporting import *

class daeVTKDataReporter(daeDataReporterLocal):
    """
    Saves data in the VTK format (.vtk) using pyEVTK module avaialable
    at https://bitbucket.org/somada141/pyevtk. Install using: "pip install pyevtk".
    Nota bene:
      It is not an original module available at https://bitbucket.org/pauloh/pyevtk.
    Does not require VTK installed.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName   = ""
        self.ConnectString = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        try:
            from pyevtk.hl import gridToVTK

            base_folder = self.ConnectString
            if not os.path.isdir(base_folder):
                os.mkdir(base_folder)

            vtk_visit = {}
            for variable_name, (ndarr_values, ndarr_times, l_domains, s_units) in self.Process.dictVariableValues.items():
                varName = daeGetStrippedName(variable_name).split('.')[-1]
                varStrippedName = daeGetStrippedName(variable_name)
                for (t,), time in numpy.ndenumerate(ndarr_times):
                    filename = '%s - %.5fs' % (varStrippedName, time)
                    filepath = os.path.join(base_folder, filename)
                    x = numpy.array([0.0])
                    y = numpy.array([0.0])
                    z = numpy.array([0.0])
                    nd = len(l_domains)
                    if nd > 3:
                        break

                    if nd == 1:
                        x = numpy.array(l_domains[0])
                    elif nd == 2:
                        x = numpy.array(l_domains[0])
                        y = numpy.array(l_domains[1])
                    elif nd == 3:
                        x = numpy.array(l_domains[0])
                        y = numpy.array(l_domains[1])
                        z = numpy.array(l_domains[2])

                    if nd == 0:
                        values = numpy.array([ndarr_values[t]])
                    else:
                        values = numpy.array(ndarr_values[t])

                    values = values.reshape((len(x), len(y), len(z)))

                    gridToVTK(filepath, x, y, z, pointData = {varName : values})
                    if not varStrippedName in vtk_visit:
                        vtk_visit[varStrippedName] = []
                    vtk_visit[varStrippedName].append(filename + '.vtr')

            for var, files in vtk_visit.items():
                filename = '%s.visit' % var
                filepath = os.path.join(base_folder, filename)
                f = open(filepath, 'w')
                f.write('\n'.join(files))
                f.close()

        except Exception as e:
            print(('Cannot write results in .vtk format:\n' + str(e)))

class daeMatlabMATFileDataReporter(daeDataReporterLocal):
    """
    Saves data in Matlab MAT format format (.mat) using scipy.io.savemat function.
    Does not need Matlab installed.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName   = ""
        self.ConnectString = ""

    def Connect(self, ConnectString, ProcessName):
        print(ConnectString, ProcessName)
        
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        mdict = {}
        for var in self.Process.Variables:
            mdict[var.Name] = var.Values

        try:
            import scipy.io
            scipy.io.savemat(self.ConnectString,
                             mdict,
                             appendmat=False,
                             format='5',
                             long_field_names=False,
                             do_compression=False,
                             oned_as='row')
                             
        except Exception as e:
            print(('Cannot write results in .mat format:\n' + str(e)))


class daeJSONFileDataReporter(daeDataReporterLocal):
    """
    Saves data in JSON text format using python json library.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        mdict = {}
        for variable_name, (ndarr_values, ndarr_times, l_domains, s_units) in self.Process.dictVariableValues.items():
            mdict[daeGetStrippedName(variable_name)] = {'Values'  : ndarr_values.tolist(),
                                                        'Times'   : ndarr_times.tolist(),
                                                        'Domains' : l_domains,
                                                        'Units'   : s_units
                                                       }

        try:
            import json
            f = open(self.ConnectString, 'w')
            f.write(json.dumps(mdict, sort_keys=True, indent=4))
            f.close()
            
        except Exception as e:
            print(('Cannot write data in JSON format:\n' + str(e)))
   
class daeHDF5FileDataReporter(daeDataReporterLocal):
    """
    Saves data in HDF5 format using python h5py library.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        try:
            import h5py
            f = h5py.File(self.ConnectString, "w")
            for variable_name, (ndarr_values, ndarr_times, l_domains, s_units) in self.Process.dictVariableValues.items():
                grp = f.create_group(daeGetStrippedName(variable_name))
                dsv = grp.create_dataset("Values",  data = ndarr_values)
                dst = grp.create_dataset("Times",   data = ndarr_times)
                dsd = grp.create_dataset("Domains", data = l_domains)
                dsu = grp.create_dataset("Units",   data = s_units)

            f.close()

        except Exception as e:
            print(('Cannot write data in HDF5 format:\n' + str(e)))

class daeXMLFileDataReporter(daeDataReporterLocal):
    """
    Saves data in XML format (.xml) using python xml library.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        try:
            from xml.etree import ElementTree

            class XmlDictObject(dict):
                """
                Adds object like functionality to the standard dictionary.
                """

                def __init__(self, initdict=None):
                    if initdict is None:
                        initdict = {}
                    dict.__init__(self, initdict)

                def __getattr__(self, item):
                    return self.__getitem__(item)

                def __setattr__(self, item, value):
                    self.__setitem__(item, value)

                def __str__(self):
                    if self.has_key('_text'):
                        return self.__getitem__('_text')
                    else:
                        return ''

                @staticmethod
                def Wrap(x):
                    """
                    Static method to wrap a dictionary recursively as an XmlDictObject
                    """

                    if isinstance(x, dict):
                        return XmlDictObject((k, XmlDictObject.Wrap(v)) for (k, v) in x.items())
                    elif isinstance(x, list):
                        return [XmlDictObject.Wrap(v) for v in x]
                    else:
                        return x

                @staticmethod
                def _UnWrap(x):
                    if isinstance(x, dict):
                        return dict((k, XmlDictObject._UnWrap(v)) for (k, v) in x.items())
                    elif isinstance(x, list):
                        return [XmlDictObject._UnWrap(v) for v in x]
                    else:
                        return x

                def UnWrap(self):
                    """
                    Recursively converts an XmlDictObject to a standard dictionary and returns the result.
                    """

                    return XmlDictObject._UnWrap(self)

            def _ConvertDictToXmlRecurse(parent, dictitem):
                assert type(dictitem) is not type([])

                if isinstance(dictitem, dict):
                    for (tag, child) in dictitem.items():
                        if str(tag) == '_text':
                            parent.text = str(child)
                        elif type(child) is type([]):
                            # iterate through the array and convert
                            for listchild in child:
                                elem = ElementTree.Element(tag)
                                parent.append(elem)
                                _ConvertDictToXmlRecurse(elem, listchild)
                        else:
                            elem = ElementTree.Element(tag)
                            parent.append(elem)
                            _ConvertDictToXmlRecurse(elem, child)
                else:
                    parent.text = str(dictitem)

            def ConvertDictToXml(xmldict):
                """
                Converts a dictionary to an XML ElementTree Element
                """

                roottag = list(xmldict.keys())[0]
                root = ElementTree.Element(roottag)
                _ConvertDictToXmlRecurse(root, xmldict[roottag])
                return root

            mdict = XmlDictObject()
            variables = {}

            for var in self.Process.Variables:
                variable = {}
                variable['Units'] = var.Units
                variable['Times'] = {}
                variable['Times']['item'] = var.TimeValues.tolist()
                # ConvertDictToXml complains about multi-dimensional arrays
                # Hence, flatten nd_array before exporting to xml
                variable['Values'] = {}
                variable['Values']['item'] = numpy.ravel(var.Values).tolist()
                variables[daeGetStrippedName(var.Name)] = variable
            mdict['Simulation'] = variables

            root = ConvertDictToXml(mdict)
            tree = ElementTree.ElementTree(root)
            tree.write(self.ConnectString)
            
        except Exception as e:
            print(('Cannot write data in XML format:\n' + str(e)))

           
class daeExcelFileDataReporter(daeDataReporterLocal):
    """
    Saves data in MS Excel format (.xls) using python xlwt library.
    Does not need Excel installed (works under GNU/Linux too).
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.WriteDataToFile()
        return True

    def WriteDataToFile(self):
        try:
            import xlwt

            wb = xlwt.Workbook()
            # Uses a new property (dictVariableValues) in daeDataReporterLocal to process the data
            for variable_name, (ndarr_values, ndarr_times, l_domains, s_units) in self.Process.dictVariableValues.items():
                ws = wb.add_sheet(variable_name)

                ws.write(0, 0, 'Times')
                ws.write(0, 1, 'Values [%s]' % s_units)
                for (t,), time in numpy.ndenumerate(ndarr_times):
                    ws.write(t+1, 0, time)
                    v = 0
                    for val_indexes, value in numpy.ndenumerate(ndarr_values[t]):
                        ws.write(t+1, v+1, value)
                        v += 1

            wb.save(self.ConnectString)
            
        except Exception as e:
            print(('Cannot write data to an excel file:\n' + str(e)))

class daePandasDataReporter(daeDataReporterLocal):
    """
    Creates pandas DataSet using pandas library.
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""
        self.data_frame  = None

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName   = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.GenerateDataSet()
        return True

    def GenerateDataSet(self):
        try:
            import pandas
            from pandas import DataFrame
            
            names = []
            data  = []
            times = []
            units = []
            for name, var in self.Process.dictVariables.items():
                names.append(name)
                units.append(var.Units)
                times.append(var.TimeValues)
                data.append(var.Values)

            _data = list(zip(units, times, data))
            self.data_frame = DataFrame(data = _data, columns = ['Units', 'Times', 'Values'], index = names)

        except Exception as e:
            print(('Cannot generate Pandas DataFrame:\n' + str(e)))

class daePlotDataReporter(daeDataReporterLocal):
    """
    Plots the specified variables using Matplotlib (by Caleb Hattingh).
    """
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName      = ProcessName
        self.ConnectString = ConnectString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        return True

    def Plot(self, *args, **kwargs):
        '''
        ``args`` can be either:

        a) Instances of daeVariable, or
        b) Lists of daeVariable instances, or
        c) A mixture of both.

        Each ``arg`` will get its own subplot. The subplots are all automatically
        arranged such that the resulting figure is as square-like as
        possible. You can however override the shape by supplying ``figRows``
        and ``figCols`` as keyword args.

        Basic Example::

            # Create Log, Solver, DataReporter and Simulation object
            log = daePythonStdOutLog()
            daesolve = daeIDAS()
            from daetools.pyDAE.data_reporters import daePlotDataReporter
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
                if kw in kwargs and kwargs[kw]:
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
                return '{0} ({1})'.format(daeVar.Name, str(daeVar.VariableType.Units))

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
                if daeVar.CanonicalName in lookup:
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
                try:
                    plt.tight_layout(1.2)
                except Exception as e:
                    pass
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

        except Exception as e:
            import traceback
            print(('Cannot generate matplotlib plots:\n' + traceback.format_exc()))
