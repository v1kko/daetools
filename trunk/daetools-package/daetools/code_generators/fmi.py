import os, shutil, sys, numpy, math, traceback, uuid, zipfile, tempfile, json, time
import daetools
from daetools.pyDAE import *
from .c99 import daeCodeGenerator_c99
from .fmi_xml_support import *

class daeCodeGenerator_FMI(fmiModelDescription):
    def __init__(self):
        fmiModelDescription.__init__(self)        
    
    def generateSimulation(self, simulation, directory, py_simulation_file, simulation_classname, py_additional_files = []):
        try:
            if not simulation:
                raise RuntimeError('Invalid simulation object')
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if not isinstance(py_additional_files, list):
                raise RuntimeError('Additional python files must be a list')
            if not simulation_classname:
                raise RuntimeError('No python simulation name specified for FMU')
            
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
            files_to_copy = py_additional_files + [py_simulation_file]
            for py_file in files_to_copy:
                py_path, py_filename = os.path.split(py_file)
                shutil.copy2(py_file, os.path.join(resources_dir, py_filename))

            # Copy all available libdaetools_fmi_cs-{platform}_{system}.[so/dll/dynlib] to the 'binaries/platform[32/64]' folder
            self._copy_solib('Linux',   'x86_64', modelIdentifier, binaries_dir)
            self._copy_solib('Linux',   'i386',   modelIdentifier, binaries_dir)
            self._copy_solib('Windows', 'win32',  modelIdentifier, binaries_dir)
            self._copy_solib('Darwin',  'x86_64', modelIdentifier, binaries_dir)
            self._copy_solib('Darwin',  'i386',   modelIdentifier, binaries_dir)

            # Generate settings.json file
            f = open(os.path.join(resources_dir, 'settings.json'), "w")
            settings = {}
            settings['simulationClass'] = simulation_classname
            settings['simulationFile']  = py_simulation_file
            if simulation.DAESolver.LASolver:
                settings['LASolver']    = simulation.DAESolver.LASolver.Name
            else:
                settings['LASolver']    = ''
            f.write(json.dumps(settings, indent = 4, sort_keys = True))
            f.close()

            # Generate initialization.json file 
            daeSimulationExplorer.saveJSONSettings(os.path.join(resources_dir, 'init.json'), simulation, simulation_classname)

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
            for ref, f in fmi_interface.items():
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
        # Copy libdaetools_fmi_cs-{platform}_{system}.[so/dll/dynlib] to the 'binaries/platform[32/64]' folder
        if platform_system == 'Linux':
            so_ext = 'so'
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'linux64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'linux32')
        elif platform_system == 'Windows':
            so_ext = 'dll'
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'win64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'win32')
            # Modify platform_machine to match daetools naming
            platform_machine = 'win32'
        elif platform_system == 'Darwin':
            so_ext = 'dynlib'
            if platform_machine == 'x86_64':
                platform_binaries_dir = os.path.join(binaries_dir, 'darwin64')
            else:
                platform_binaries_dir = os.path.join(binaries_dir, 'darwin32')
            # Modify platform_machine to match daetools naming
            platform_machine = 'universal'
        else:
            raise RuntimeError('Unsupported platform: %s' % platform_system)

        daetools_fmu_solib = '%s.%s' % (modelIdentifier, so_ext)
        daetools_fmi_cs = 'libdaetools_fmi_cs-%s_%s.%s' % (platform_system, platform_machine, so_ext)
        #print daetools_fmi_cs
        try:
            shutil.copy2(os.path.join(daetools.daetools_dir, 'code_generators', 'fmi', daetools_fmi_cs),
                         os.path.join(platform_binaries_dir, daetools_fmu_solib))
        except Exception as e:
            # Ignore exceptions, since some of binaries could be missing
            pass
        
    def _addInput(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityInput
        sv.variability    = fmiScalarVariable.variabilityContinuous # Set it to continuous (page 49 FMI-v2.0.pdf)
        sv.initial        = fmiScalarVariable.initialExact
        sv.type           = fmiReal()
        sv.type.declaredType = fmi_obj.variable.VariableType.Name
        sv.type.start        = fmi_obj.variable.GetValue(list(fmi_obj.indexes))
        self.ModelVariables.append(sv)

    def _addOutput(self, fmi_obj):
        #
        unknown = fmiVariableDependency()
        unknown.index = int(fmi_obj.reference) #*
        self.ModelStructure.Outputs.append(unknown)

        unknown = fmiVariableDependency()
        unknown.index = int(fmi_obj.reference) #*
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
        sv.type.start     = fmi_obj.parameter.GetValue()
        self.ModelVariables.append(sv)

    def _addSTN(self, fmi_obj):
        sv = fmiScalarVariable()
        sv.name           = str(fmi_obj.name) #*
        sv.valueReference = int(fmi_obj.reference) #*
        sv.description    = str(fmi_obj.description)
        sv.causality      = fmiScalarVariable.causalityInput # If it is input then it can be changed by the simulator
        sv.variability    = fmiScalarVariable.variabilityDiscrete
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
