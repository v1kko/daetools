import os, shutil, sys, numpy, math, traceback, uuid
from daetools.pyDAE import *
from ansi_c import daeCodeGenerator_ANSI_C
from fmi_xml_support import *

class daeCodeGenerator_FMI(fmiModelDescription):
    def __init__(self):
        fmiModelDescription.__init__(self)        
    
    def generateSimulation(self, simulation, **kwargs):
        directory  = kwargs.get('projectDirectory', None)
        filename   = os.path.join(directory, 'modelDescription.fmu')
        source_dir = os.path.join(directory, 'sources')

        folder = directory
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'documentation')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'sources')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'resources')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/win32')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/win64')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/linux32')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/linux64')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/darwin32')
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(directory, 'binaries/darwin64')
        if not os.path.exists(folder):
            os.makedirs(folder)

        cgANSI_C = daeCodeGenerator_ANSI_C()
        cgANSI_C.generateSimulation(simulation, projectDirectory = source_dir)

        self.modelName                  = '' #*
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
        self.CoSimulation.modelIdentifier                        = '' #*
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
        self.VendorAnnotations = []   # [fmiVendorAnnotation()]
        self.DefaultExperiment = None # fmiDefaultExperiment()

        self._addParameter('Param', 0, '')

        self.to_xml(filename)

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
