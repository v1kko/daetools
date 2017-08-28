"""
***********************************************************************************
                            fmi.py
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
import os, shutil, sys, numpy, math, traceback, uuid, zipfile, tempfile, json, time, glob
import daetools
from daetools.pyDAE import *
from .c99 import daeCodeGenerator_c99
from .fmi_xml_support import *

class daeCodeGenerator_FMI(fmiModelDescription):
    def __init__(self):
        fmiModelDescription.__init__(self)        
    
    def generateSimulation(self, simulation, 
                                 directory, 
                                 py_simulation_file, 
                                 callable_object_name, 
                                 arguments, 
                                 additional_files = [], 
                                 localsAsOutputs = True):
        try:
            tmp_folder = ''
            if not simulation:
                raise RuntimeError('Invalid simulation object')
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if not isinstance(additional_files, list):
                raise RuntimeError('Additional python files must be a list of tuples')
            if not callable_object_name:
                raise RuntimeError('No python callable object name specified for FMU')
            
            self.localsAsOutputs     = localsAsOutputs
            self.wrapperInstanceName = simulation.m.Name
            modelIdentifier          = simulation.m.GetStrippedName()
            
            fmu_directory            = directory
            tmp_folder               = tempfile.mkdtemp(prefix = 'daetools-fmu-')
            xml_description_filename = os.path.join(tmp_folder, 'modelDescription.xml')
            sources_dir              = os.path.join(tmp_folder, 'sources')
            resources_dir            = os.path.join(tmp_folder, 'resources')
            binaries_dir             = os.path.join(tmp_folder, 'binaries')
            fmu_filename             = os.path.join(fmu_directory, simulation.m.Name + '.fmu')

            folder = tmp_folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'documentation')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'sources')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'resources')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/win32')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/win64')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/linux32')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/linux64')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/darwin32')
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = os.path.join(tmp_folder, 'binaries/darwin64')
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Copy the python file with the simulation class and all additional files to the 'resources' folder
            # py_simulation_file argument is often specified using the __file__ attribute.
            # In Python 2.7 __file__ points to 'file.pyc' - correct it to 'file.py'
            if py_simulation_file.endswith('.pyc'):
                py_simulation_file = py_simulation_file[:-1]

            py_path, py_filename = os.path.split(str(py_simulation_file))
            shutil.copy2(py_simulation_file, os.path.join(resources_dir, py_filename))
            
            # Copy the additional files to the locations relative to the 'resources' folder
            # Additional files are a list of tuples: [('file_path', 'resources_dir_relative_path'), ...]
            for py_file, relative_path in additional_files:
                destination_file = os.path.join(resources_dir, relative_path)
                py_path, py_filename = os.path.split(destination_file)
                if not os.path.isdir(py_path):
                    os.makedirs(py_path)
                shutil.copy2(py_file, destination_file)

            # Copy all available libcdaeFMU_CS-pyXY.[so/dll/dynlib] to the 'binaries/platform[32/64]' folder
            self._copy_solib('Linux',   'x86_64', modelIdentifier, binaries_dir)
            self._copy_solib('Linux',   'i386',   modelIdentifier, binaries_dir)
            self._copy_solib('Windows', 'win32',  modelIdentifier, binaries_dir)
            self._copy_solib('Darwin',  'x86_64', modelIdentifier, binaries_dir)
            self._copy_solib('Darwin',  'i386',   modelIdentifier, binaries_dir)

            # Generate settings.json file
            f = open(os.path.join(resources_dir, 'settings.json'), "w")
            settings = {}
            settings['simulationFile']     = os.path.basename(py_simulation_file)
            settings['callableObjectName'] = callable_object_name
            settings['arguments']          = arguments
            f.write(json.dumps(settings, indent = 4, sort_keys = True))
            f.close()

            # Generate initialization.json file 
            # daeSimulationExplorer.saveJSONSettings(os.path.join(resources_dir, 'init.json'), simulation, callable_object_name)

            # Fill in the xml data
            self.modelName                  = modelIdentifier #*
            self.guid                       = uuid.uuid1() #*
            self.description                = simulation.m.Description
            self.author                     = ''
            self.version                    = ''
            self.copyright                  = ''
            self.license                    = ''
            self.generationTool             = 'DAE Tools v%s' % daeVersion(True)
            self.generationDateAndTime      = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())
            self.variableNamingConvention   = fmiModelDescription.variableNamingConventionStructured
            self.numberOfEventIndicators    = 0

            self.CoSimulation = fmiCoSimulation()
            self.CoSimulation.modelIdentifier                        = modelIdentifier #*
            self.CoSimulation.needsExecutionTool                     = True
            self.CoSimulation.canHandleVariableCommunicationStepSize = True
            self.CoSimulation.canHandleEvents                        = True
            self.CoSimulation.canInterpolateInputs                   = False
            self.CoSimulation.maxOutputDerivativeOrder               = 0
            self.CoSimulation.canRunAsynchronuously                  = False
            self.CoSimulation.canSignalEvents                        = True
            self.CoSimulation.canBeInstantiatedOnlyOncePerProcess    = False
            self.CoSimulation.canNotUseMemoryManagementFunctions     = True
            self.CoSimulation.canGetAndSetFMUstate                   = False
            self.CoSimulation.canSerializeFMUstate                   = False
            self.CoSimulation.providesPartialDerivativesOf_DerivativeFunction_wrt_States = False
            self.CoSimulation.providesPartialDerivativesOf_DerivativeFunction_wrt_Inputs = False
            self.CoSimulation.providesPartialDerivativesOf_OutputFunction_wrt_States     = False
            self.CoSimulation.providesPartialDerivativesOf_OutputFunction_wrt_Inputs     = False

            self.ModelStructure    = fmiModelStructure()
            self.ModelVariables    = []
            self.UnitDefinitions   = []
            self.TypeDefinitions   = []
            self.VendorAnnotations = []
            self.DefaultExperiment = fmiDefaultExperiment()
            self.LogCategories     = []

            # Setup a default experiment
            self.DefaultExperiment.startTime = 0.0
            self.DefaultExperiment.stopTime  = simulation.TimeHorizon
            self.DefaultExperiment.tolerance = simulation.DAESolver.RelativeTolerance

            # Add model variables
            fmi_interface = simulation.m.GetFMIInterface()
            #print fmi_interface

            variableTypesUsed = {}
            unitsUsed = {}
            for ref, f in sorted(fmi_interface.items()):
                if f.type == 'Input':
                    self._addInput(f)
                    if not f.variable.VariableType.Name in variableTypesUsed:
                        variableTypesUsed[f.variable.VariableType.Name] = f.variable.VariableType
                    if not str(f.variable.VariableType.Units) in unitsUsed:
                        unitsUsed[str(f.variable.VariableType.Units)] = f.variable.VariableType.Units

                elif f.type == 'Output':
                    self._addOutput(f)
                    if not f.variable.VariableType.Name in variableTypesUsed:
                        variableTypesUsed[f.variable.VariableType.Name] = f.variable.VariableType
                    if not str(f.variable.VariableType.Units) in unitsUsed:
                        unitsUsed[str(f.variable.VariableType.Units)] = f.variable.VariableType.Units

                elif f.type == 'Local':
                    # Treat all locals as outputs at the moment
                    if self.localsAsOutputs:
                        self._addOutput(f)
                    else:
                        self._addLocal(f)
                    if not f.variable.VariableType.Name in variableTypesUsed:
                        variableTypesUsed[f.variable.VariableType.Name] = f.variable.VariableType
                    if not str(f.variable.VariableType.Units) in unitsUsed:
                        unitsUsed[str(f.variable.VariableType.Units)] = f.variable.VariableType.Units

                elif f.type == 'Parameter':
                    self._addParameter(f)
                    if not str(f.parameter.Units) in unitsUsed:
                        unitsUsed[str(f.parameter.Units)] = f.parameter.Units

                elif f.type == 'STN':
                    self._addSTN(f)

                else:
                    raise RuntimeError('Invalid variable reference type')

            # Add unit definitions
            for unit_name, u in unitsUsed.items():
                self._addUnitDefinition(u)

            # Add variable types
            for vartype_name, var_type in variableTypesUsed.items():
                self._addTypeDefinition(var_type)

            # Add log categories
            cat = fmiLogCategory()
            cat.name = 'logAll'
            self.LogCategories.append(cat)

            # Save model description xml file
            self.to_xml(xml_description_filename)

            files_to_zip = []
            for root, dirs, files in os.walk(tmp_folder):
                for file in files:
                    filename      = os.path.join(root, file)
                    relative_path = os.path.relpath(filename, tmp_folder)
                    files_to_zip.append( (filename, relative_path) )

            zip = zipfile.ZipFile(fmu_filename, "w")
            for filename, relative_path in files_to_zip:
                zip.write(filename, relative_path)
            zip.close()

        finally:
            # Remove temporary directory
            if os.path.isdir(tmp_folder):
                #shutil.rmtree(tmp_folder)
                pass

    def _formatUnits(self, units):
        # Format: m.kg2/s-2 meaning m * kg**2 / s**2
        positive = []
        negative = []
        for u, exp in list(units.toDict().items()):
            if exp >= 0:
                if exp == 1:
                    positive.append('{0}'.format(u))
                elif int(exp) == exp:
                    positive.append('{0}^{1}'.format(u, int(exp)))
                else:
                    positive.append('{0}^{1}'.format(u, exp))

        for u, exp in list(units.toDict().items()):
            if exp < 0:
                if exp == -1:
                    negative.append('{0}'.format(u))
                elif int(exp) == exp:
                    negative.append('{0}^{1}'.format(u, int(math.fabs(exp))))
                else:
                    negative.append('{0}^{1}'.format(u, math.fabs(exp)))

        if len(positive) == 0:
            sPositive = 'rad'
        else:
            sPositive = '.'.join(positive)

        if len(negative) == 0:
            sNegative = ''
        elif len(negative) == 1:
            sNegative = '/' + '.'.join(negative)
        else:
            sNegative = '/(' + '.'.join(negative) + ')'

        return sPositive + sNegative
        
    def _copy_solib(self, platform_system, platform_machine, modelIdentifier, binaries_dir):
        # Copy libcdaeFMU_CS-pyXY.[so/dll/dynlib] and libcdaeSimulationLoader-pyXY.[so/dll/dynlib]
        # to the 'binaries/platform[32/64]' folder
        if platform_system == 'Linux':
            so_ext = 'so'
            so_ext_pattern = 'so.*'
            shared_lib_prefix = 'lib'
            shared_lib_postfix = ''
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'linux64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'linux32')
        elif platform_system == 'Windows':
            so_ext = 'dll'
            so_ext_pattern = 'dll'
            shared_lib_prefix = ''
            shared_lib_postfix = '1'
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'win64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'win32')
            # Modify platform_machine to match daetools naming
            platform_machine = 'win32'
        elif platform_system == 'Darwin':
            so_ext = 'dylib'
            so_ext_pattern = 'dylib'
            shared_lib_prefix = 'lib'
            shared_lib_postfix = ''
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'darwin64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'darwin32')
            # Modify platform_machine to match daetools naming
            platform_machine = 'universal'
        else:
            raise RuntimeError('Unsupported platform: %s' % platform_system)

        solibs_dir = os.path.join(daetools.daetools_dir, 'solibs', '%s_%s' % (platform_system, platform_machine))
        daetools_fmu_solib = '%s.%s' % (modelIdentifier, so_ext)
        daetools_fmi_cs = '%scdaeFMU_CS-py%s%s%s.%s' % (shared_lib_prefix,
                                                        daetools.python_version_major,
                                                        daetools.python_version_minor,
                                                        shared_lib_postfix,
                                                        so_ext)
        daetools_simulation_loader = '%scdaeSimulationLoader-py%s%s%s.%s' % (shared_lib_prefix,
                                                                             daetools.python_version_major,
                                                                             daetools.python_version_minor,
                                                                             shared_lib_postfix,
                                                                             so_ext)
        boost_files = glob.iglob(os.path.join(solibs_dir, "*boost_*-daetools-py%s%s.%s" % (daetools.python_version_major,
                                                                                           daetools.python_version_minor,
                                                                                           so_ext_pattern)))

        try:
            # Copy FMU_CS
            _source = os.path.join(solibs_dir,            daetools_fmi_cs)
            _target = os.path.join(platform_binaries_dir, daetools_fmu_solib)
            #print('copy %s %s' % (_source, _target))
            shutil.copy2(_source, _target)
        except Exception as e:
            # Ignore exceptions, since some of binaries are certainly not available
            pass

        try:
            # Copy SimulationLoader
            _source = os.path.join(solibs_dir, daetools_simulation_loader)
            _target = os.path.join(platform_binaries_dir)
            #print('copy %s %s' % (_source, _target))
            shutil.copy2(_source, _target)
        except Exception as e:
            # Ignore exceptions, since some of binaries are certainly not available
            pass

        try:
            # Copy boost libs
            _target = os.path.join(platform_binaries_dir)
            for _source in boost_files:
                if os.path.isfile(_source):
                    #print('copy %s %s' % (_source, _target))
                    shutil.copy2(_source, _target)
        except Exception as e:
            # Ignore exceptions, since some of binaries are certainly not available
            pass

    def _addInput(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityInput
        sv.variability    = fmiScalarVariable.variabilityContinuous # Set it to continuous (page 49 FMI-v2.0.pdf)
        sv.initial        = None # It is not allowed to provide a value for initial if causality = "input" or "independent"
        sv.type           = fmiReal()
        sv.type.declaredType = fmi_obj.variable.VariableType.Name
        sv.type.start        = fmi_obj.variable.GetValue(list(fmi_obj.indexes))
        self.ModelVariables.append(sv)

    def _addOutput(self, fmi_obj):
        # In general, the index is not equal to he reference but to the index in the ModelVariables list.
        # But, the references in daetools start at 1 and increase and they are added in the sorted order.
        # So it should be fine to use reference as an index.
        # Anyway, the index in the ModelStructure.Outputs is used.
        # The indexes in FMI start at 1 (not at zero).
        var_index = len(self.ModelVariables) + 1
        
        unknown = fmiVariableDependency()
        unknown.index = var_index #*
        self.ModelStructure.Outputs.append(unknown)

        unknown = fmiVariableDependency()
        unknown.index = var_index #*
        self.ModelStructure.InitialUnknowns.append(unknown)
        
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityOutput
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        sv.type           = fmiReal()
        sv.type.declaredType = fmi_obj.variable.VariableType.Name
        self.ModelVariables.append(sv)

    def _addLocal(self, fmi_obj):
        # Here, do not add anything in the ModelStructure
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        sv.type           = fmiReal()
        sv.type.declaredType = fmi_obj.variable.VariableType.Name
        self.ModelVariables.append(sv)

    def _addParameter(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityTunable # If it is tunable it can be changed during the simulation
        sv.initial        = fmiScalarVariable.initialExact
        sv.type           = fmiReal()
        sv.type.unit      = self._formatUnits(fmi_obj.parameter.Units)
        sv.type.start     = fmi_obj.parameter.GetValue(list(fmi_obj.indexes))
        self.ModelVariables.append(sv)

    def _addSTN(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityParameter # If it is input then it can be changed by the simulator
        sv.variability    = fmiScalarVariable.variabilityTunable
        sv.initial        = fmiScalarVariable.initialExact
        sv.type           = fmiString()
        sv.type.start     = fmi_obj.stn.ActiveState
        self.ModelVariables.append(sv)
        
    """
    def _addNumberOfPointsInDomain(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityConstant
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addDomainPoints(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityTunable
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addAssignedVariable(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityTunable
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addAlgebraicVariable(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)

    def _addDifferentialVariable(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)

    def _addInletPortVariable(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityInput
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addOutletPortVariable(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description) + ' [%s]' % fmi_obj.units
        sv.causality      = fmiScalarVariable.causalityOutput
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)
    """
    
    def _addUnitDefinition(self, dae_unit):
        dae_bu = dae_unit.baseUnit
        
        unit = fmiUnit()
        unit.name = self._formatUnits(dae_unit) # *
        
        unit.baseUnit = fmiBaseUnit()
        unit.baseUnit.factor = dae_bu.multiplier
        unit.baseUnit.offset = 0.0
        if dae_bu.L != 0:
            unit.baseUnit.m = dae_bu.L
        if dae_bu.M != 0:
            unit.baseUnit.kg = dae_bu.M
        if dae_bu.T != 0:
            unit.baseUnit.s = dae_bu.T
        if dae_bu.C != 0:
            unit.baseUnit.cd = dae_bu.C
        if dae_bu.I != 0:
            unit.baseUnit.A = dae_bu.I
        if dae_bu.O != 0:
            unit.baseUnit.K = dae_bu.O
        if dae_bu.N != 0:
            unit.baseUnit.mol = dae_bu.N

        # If all are 0, set rad = 1
        if dae_bu.L == 0 and dae_bu.M == 0 and dae_bu.T == 0 and dae_bu.C == 0 and dae_bu.I == 0 and dae_bu.O == 0 and dae_bu.N == 0:
            unit.baseUnit.rad = 1

        self.UnitDefinitions.append(unit)

    def _addTypeDefinition(self, var_type):
        st = fmiSimpleType()
        st.name = var_type.Name #*
 
        st.type = fmiReal()
        st.type.unit    = self._formatUnits(var_type.Units)
        st.type.min     = var_type.LowerBound
        st.type.max     = var_type.UpperBound
        st.type.nominal = var_type.InitialGuess

        self.TypeDefinitions.append(st)
