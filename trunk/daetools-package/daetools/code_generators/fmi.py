import os, shutil, sys, numpy, math, traceback, uuid, zipfile, tempfile
from daetools.pyDAE import *
from ansi_c import daeCodeGenerator_ANSI_C
from fmi_xml_support import *

class daeCodeGenerator_FMI(fmiModelDescription):
    def __init__(self):
        fmiModelDescription.__init__(self)        
    
    def generateSimulation(self, simulation, directory):
        try:
            if not simulation:
                raise RuntimeError('Invalid simulation object')
            if not os.path.isdir(directory):
                os.makedirs(directory)
            
            fmu_directory = directory
            
            tmp_folder               = tempfile.mkdtemp(prefix = 'daetools-fmu-')
            xml_description_filename = os.path.join(tmp_folder, 'modelDescription.xml')
            source_dir               = os.path.join(tmp_folder, 'sources')
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

            cgANSI_C = daeCodeGenerator_ANSI_C()
            cgANSI_C.generateSimulation(simulation, projectDirectory = source_dir)
            self.wrapperInstanceName = simulation.m.Name

            self.modelName                  = simulation.m.Name #*
            self.guid                       = uuid.uuid1() #*
            self.description                = ''
            self.author                     = ''
            self.version                    = ''
            self.copyright                  = ''
            self.license                    = ''
            self.generationTool             = 'DAE Tools'
            self.generationDateAndTime      = ''
            self.variableNamingConvention   = fmiModelDescription.variableNamingConventionStructured
            self.numberOfEventIndicators    = 0

            self.CoSimulation = fmiCoSimulation()
            self.CoSimulation.modelIdentifier                        = simulation.m.Name #*
            self.CoSimulation.needsExecutionTool                     = False
            self.CoSimulation.canHandleVariableCommunicationStepSize = True
            self.CoSimulation.canHandleEvents                        = True
            self.CoSimulation.canInterpolateInputs                   = False
            self.CoSimulation.maxOutputDerivativeOrder               = 1
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

            # Setup a default experiment
            self.DefaultExperiment.startTime = 0.0
            self.DefaultExperiment.stopTime  = simulation.TimeHorizon
            self.DefaultExperiment.tolerance = simulation.DAESolver.RelativeTolerance

            # Add unit definitions

            # Add variable types
            
            # Add model structure (inputs/outputs)
            
            # Add model variables
            for i, (ref_type, ref_name, ref_flat_name, block_index) in enumerate(cgANSI_C.floatValuesReferences):
                if ref_type == 'Assigned':
                    self._addAssignedVariable(ref_name, i, '')

                elif ref_type == 'Algebraic':
                    self._addAlgebraicVariable(ref_name, i, '')

                elif ref_type == 'Differential':
                    self._addDifferentialVariable(ref_name, i, '')

                elif ref_type == 'Parameter':
                    self._addParameter(ref_name, i, '')

                elif ref_type == 'NumberOfPointsInDomain':
                    self._addNumberOfPointsInDomain(ref_name, i, '')

                elif ref_type == 'DomainPoints':
                    self._addDomainPoints(ref_name, i, '')

                else:
                    raise RuntimeError('Invalid variable reference type')

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

    def _addInput(self, name, value_ref):
        i = fmiInput()
        i.name       = str(name) #*
        i.derivative = int(value_ref)
        self.ModelStructure.Inputs.append(i)

    def _addOutput(self, name, value_ref):
        o = fmiOutput()
        o.name       = str(name) #*
        o.derivative = int(value_ref)
        self.ModelStructure.Outputs.append(o)

    def _addParameter(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityFixed
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addNumberOfPointsInDomain(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityConstant
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addDomainPoints(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityTunable
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addAssignedVariable(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityTunable
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addAlgebraicVariable(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)

    def _addDifferentialVariable(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityLocal
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)

    def _addSTN(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityParameter
        sv.variability    = fmiScalarVariable.variabilityDiscrete
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addInletPortVariable(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityInput
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialExact
        self.ModelVariables.append(sv)

    def _addOutletPortVariable(self, name, value_ref, description):
        sv = fmiScalarVariable()
        sv.name           = str(name) #*
        sv.valueReference = int(value_ref) #*
        sv.description    = str(description)
        sv.causality      = fmiScalarVariable.causalityOutput
        sv.variability    = fmiScalarVariable.variabilityContinuous
        sv.initial        = fmiScalarVariable.initialCalculated
        self.ModelVariables.append(sv)

    def _addUnitDefinition(self, dae_unit):
        dae_bu = dae_unit.baseUnit
        
        unit = fmiUnit()
        bu = fmiBaseUnit()
        unit.name     = str(dae_unit) # *
        unit.baseUnit = bu

        bu.factor = dae_bu.multiplier
        bu.offset = 0.0
        if dae_bu.L != 0:
            bu.m = dae_bu.L
        if dae_bu.M != 0:
            bu.kg = dae_bu.M
        if dae_bu.T != 0:
            bu.s = dae_bu.T
        if dae_bu.C != 0:
            bu.cd = dae_bu.C
        if dae_bu.I != 0:
            bu.A = dae_bu.I
        if dae_bu.O != 0:
            bu.K = dae_bu.O
        if dae_bu.N != 0:
            bu.mol = dae_bu.N
        bu.rad = None
        self.UnitDefinitions.append(unit)

    def _addTypeDefinition(self, var_type):
        t = fmiReal()
        real = fmiSimpleType()
        real.name        = var_type.Name #*
        real.description = None
        real.type        = t #*

        t.quantity           = None
        t.unit               = str(var_type.Unit)
        t.displayUnit        = None
        t.relativeQuantity   = True
        t.min                = var_type.LB
        t.max                = var_type.UB
        t.nominal            = None
        t.unbounded          = True

        self.TypeDefinitions.append(real)
