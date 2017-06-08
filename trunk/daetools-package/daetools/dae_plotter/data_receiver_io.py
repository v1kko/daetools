"""********************************************************************************
                            data_receiver_io.py
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
import sys, numpy, pickle
from os.path import join, realpath, dirname
from daetools.pyDAE import *

class dataReceiverDomain(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 1: 
            # Create new from a daeDataReceiverDomain or dataReceiverDomain object
            domain = args[0]
            self.Name = domain.Name
            self.Type = domain.Type
            self.NumberOfPoints = domain.NumberOfPoints
            self.Units = domain.Units
            if domain.Type == eUnstructuredGrid:
                self.Points = []
                self.Coordinates = domain.Coordinates        
            else:
                self.Points = domain.Points
                self.Coordinates = []        
        elif len(args) == 6: 
            # Unpickling from the tuple of values
            Name, Type, NumberOfPoints, Units, Points, Coordinates = args
            self.Name = Name
            self.Type = Type
            self.NumberOfPoints = NumberOfPoints
            self.Units = Units
            self.Points = Points
            self.Coordinates = Coordinates
        else:
            raise RuntimeError('Invalid number of arguments specified in the dataReceiverDomain.__init__ function')
    
    # Pickling support
    # __reduce__ function should return a tuple (callable, tuple_with_arguments_for_the_callable)
    # This tuple will be saved into a pickle file and later used to unpickle the object.
    def __reduce__(self):
        return (self.__class__, (self.Name, self.Type, self.NumberOfPoints, self.Units, self.Points, self.Coordinates))
    
    def __repr__(self):
        return 'dataReceiverDomain(%s, Type=%d, noPoints=%d, Units=%s, Points=%s, Coords=%s)' % (self.Name, self.Type, self.NumberOfPoints, self.Units, self.Points, self.Coordinates)
    
class dataReceiverVariable(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 1: 
            # Create new from a daeDataReceiverVariable or dataReceiverVariable object
            variable = args[0]
            self.Name = variable.Name
            self.NumberOfPoints = variable.NumberOfPoints
            self.Units = variable.Units
            self.Domains = [dataReceiverDomain(dom) for dom in variable.Domains]
            self.TimeValues = variable.TimeValues
            self.Values = variable.Values
        elif len(args) == 6: 
            # Unpickling from the tuple of values
            Name, NumberOfPoints, Units, Domains, TimeValues, Values = args
            self.Name = Name
            self.NumberOfPoints = NumberOfPoints
            self.Units = Units
            self.Domains = Domains
            self.TimeValues = TimeValues
            self.Values = Values
        else:
            raise RuntimeError('Invalid number of arguments specified in the dataReceiverVariable.__init__ function')
        
    def __reduce__(self):
        return (self.__class__, (self.Name, self.NumberOfPoints, self.Units, self.Domains, self.TimeValues, self.Values))
    
    def __repr__(self):
        return 'dataReceiverVariable(%s, noPoints=%d, Units=%s, Domains=%s, TimeValues=%s, Values=%s)' % (self.Name, self.NumberOfPoints, self.Units, self.Domains, self.TimeValues, self.Values)
    
class dataReceiverProcess(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 1: 
            # Create new from a daeDataReceiverProcess or dataReceiverProcess object
            process = args[0]
            self.Name = process.Name
            self.Domains = [dataReceiverDomain(dom) for dom in process.Domains]
            self.Variables = [dataReceiverVariable(var) for var in process.Variables]
        elif len(args) == 3: 
            # Unpickling from the tuple of values
            Name, Domains, Variables = args
            self.Name = Name
            self.Domains = Domains
            self.Variables = Variables
        else:
            raise RuntimeError('Invalid number of arguments specified in the dataReceiverProcess.__init__ function')
        
    def __reduce__(self):
        return (self.__class__, (self.Name, self.Domains, self.Variables))
    
    def get_dictDomains(self):
        domains = {}
        for domain in self.Domains:
            domains[domain.Name] = domain
        return domains
    
    def get_dictVariables(self):
        variables = {}
        for variable in self.Variables:
            variables[variable.Name] = variable
        return variables
    
    def get_dictVariableValues(self):
        variableValues = {}
        for variable in self.Variables:
            variableValues[variable.Name] = (variable.Values, variable.TimeValues, [domain.Points for domain in self.Domains], variable.Units)
        return variableValues
    
    dictDomains = property(get_dictDomains)
    dictVariables = property(get_dictVariables)
    dictVariableValues = property(get_dictVariableValues)
    
    def __repr__(self):
        return 'dataReceiverProcess(\n  Name%s\n  Domains=%s\n  Variables=%s\n)' % (self.Name, self.Domains, self.Variables)


def pickleProcess(process, filename):
    data_process = dataReceiverProcess(process)
    #print(str(data_process))
    
    f = open(filename, 'wb')
    pickle.dump(data_process, f, protocol=-1)
    f.close()

def unpickleProcess(filename):
    f = open(filename, 'rb')
    data_process = pickle.load(f)
    f.close()
    
    #print(str(data_process.dictVariableValues))
    
    return data_process
    
