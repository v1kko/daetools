from lxml import etree

def addAttribute(element, name, value, required = False):
    if value != None:
        element.set(name, str(value))
    elif value == None and required:
        raise RuntimeError('Attribute [{0}] in [{1}] is required but not set'.format(name, element.tag))
  
def addElement(element, name):
    new_element = etree.SubElement(element, name)
    
def addObject(element, obj, required = False):
    # Do not write element if the object is None and element not required
    if not obj and required:
        raise RuntimeError('Object in element [{0}] is required but not set'.format(element.tag))

    if obj:
        obj_elem = etree.SubElement(element, obj.xmlTagName)
        obj.to_xml(obj_elem)

def addObjects(element, name, objects, required = False):
    if not objects and required:
        raise RuntimeError('Element [{0}] in [{1}] is required but not set'.format(name, element.tag))

    # Do not write element if the array objects is empty and element not required
    if len(objects) == 0 and not required:
        return
        
    parent_element = etree.SubElement(element, name)
    for obj in objects:
        obj_elem = etree.SubElement(parent_element, obj.xmlTagName)
        obj.to_xml(obj_elem)

class fmiModelDescription(object):
    xmlTagName = 'fmiModelDescription'

    variableNamingConventionFlat       = 'flat'
    variableNamingConventionStructured = 'structured'

    def __init__(self):
        # Required attributes
        self.fmiVersion        = "2.0" # string
        self.modelName         = ''    # string
        self.guid              = ''    # string

        # Required elements
        self.ModelStructure    = None # fmiModelStructure
        self.ModelVariables    = []   # array of fmiScalarVariable
        self.CoSimulation      = None # fmiCoSimulation

        # Optional attributes
        self.description                = None # string
        self.author                     = None # string
        self.version                    = None # string
        self.copyright                  = None # string
        self.license                    = None # string
        self.generationTool             = None # string
        self.generationDateAndTime      = None # string
        self.variableNamingConvention   = None # enum
        self.numberOfEventIndicators    = None # int

        # Optional elements
        self.DefaultExperiment = None        
        self.TypeDefinitions   = []
        self.UnitDefinitions   = []
        self.LogCategories     = []
        self.VendorAnnotations = []
    
    def to_xml(self, filename):
        fmi_root = etree.Element(self.xmlTagName)
        addAttribute(fmi_root, 'fmiVersion',                 self.fmiVersion, required = True)
        addAttribute(fmi_root, 'modelName',                  self.modelName, required = True)
        addAttribute(fmi_root, 'guid',                       self.guid, required = True)

        addAttribute(fmi_root, 'description',                self.description)
        addAttribute(fmi_root, 'author',                     self.author)
        addAttribute(fmi_root, 'version',                    self.version)
        addAttribute(fmi_root, 'copyright',                  self.copyright)
        addAttribute(fmi_root, 'license',                    self.license)
        addAttribute(fmi_root, 'generationTool',             self.generationTool)
        addAttribute(fmi_root, 'generationDateAndTime',      self.generationDateAndTime)
        addAttribute(fmi_root, 'variableNamingConvention',   self.variableNamingConvention)
        addAttribute(fmi_root, 'numberOfEventIndicators',    self.numberOfEventIndicators)

        addObject(fmi_root, self.CoSimulation, required = True)
        addObject(fmi_root, self.ModelStructure, required = True)
        addObject(fmi_root, self.DefaultExperiment)

        addObjects(fmi_root, 'TypeDefinitions',   self.TypeDefinitions)
        addObjects(fmi_root, 'UnitDefinitions',   self.UnitDefinitions)
        addObjects(fmi_root, 'LogCategories',     self.LogCategories)
        addObjects(fmi_root, 'VendorAnnotations', self.VendorAnnotations)
        addObjects(fmi_root, 'ModelVariables',    self.ModelVariables, required = True)
        
        etree.ElementTree(fmi_root).write(filename, encoding="utf-8", pretty_print=True, xml_declaration=True)
        
    @classmethod
    def from_xml(cls, filename):
        pass

class fmiCoSimulation(object):
    xmlTagName = 'CoSimulation'

    def __init__(self):
        self.modelIdentifier                        = None # Required: string
        self.needsExecutionTool                     = None # bool
        self.canHandleVariableCommunicationStepSize = None # bool
        self.canHandleEvents                        = None # bool
        self.canInterpolateInputs                   = None # bool
        self.maxOutputDerivativeOrder               = None # unsigned int
        self.canRunAsynchronuously                  = None # bool
        self.canSignalEvents                        = None # bool
        self.canBeInstantiatedOnlyOncePerProcess    = None # bool
        self.canNotUseMemoryManagementFunctions     = None # bool
        self.canGetAndSetFMUstate                   = None # bool
        self.canSerializeFMUstate                   = None # bool
        self.providesPartialDerivativesOf_DerivativeFunction_wrt_States = None # bool
        self.providesPartialDerivativesOf_DerivativeFunction_wrt_Inputs = None # bool
        self.providesPartialDerivativesOf_OutputFunction_wrt_States     = None # bool
        self.providesPartialDerivativesOf_OutputFunction_wrt_Inputs     = None # bool
        
    def to_xml(self, tag):
        addAttribute(tag, 'modelIdentifier',                        self.modelIdentifier, required = True)
        addAttribute(tag, 'needsExecutionTool',                     self.needsExecutionTool)
        addAttribute(tag, 'canHandleVariableCommunicationStepSize', self.canHandleVariableCommunicationStepSize)
        addAttribute(tag, 'canHandleEvents',                        self.canHandleEvents)
        addAttribute(tag, 'canInterpolateInputs',                   self.canInterpolateInputs)
        addAttribute(tag, 'maxOutputDerivativeOrder',               self.maxOutputDerivativeOrder)
        addAttribute(tag, 'canRunAsynchronuously',                  self.canRunAsynchronuously)
        addAttribute(tag, 'canSignalEvents',                        self.canSignalEvents)
        addAttribute(tag, 'canBeInstantiatedOnlyOncePerProcess',    self.canBeInstantiatedOnlyOncePerProcess)
        addAttribute(tag, 'canNotUseMemoryManagementFunctions',     self.canNotUseMemoryManagementFunctions)
        addAttribute(tag, 'canGetAndSetFMUstate',                   self.canGetAndSetFMUstate)
        addAttribute(tag, 'canSerializeFMUstate',                   self.canSerializeFMUstate)
        addAttribute(tag, 'providesPartialDerivativesOf_DerivativeFunction_wrt_States', self.providesPartialDerivativesOf_DerivativeFunction_wrt_States)
        addAttribute(tag, 'providesPartialDerivativesOf_DerivativeFunction_wrt_Inputs', self.providesPartialDerivativesOf_DerivativeFunction_wrt_Inputs)
        addAttribute(tag, 'providesPartialDerivativesOf_OutputFunction_wrt_States',     self.providesPartialDerivativesOf_OutputFunction_wrt_States)
        addAttribute(tag, 'providesPartialDerivativesOf_OutputFunction_wrt_Inputs',     self.providesPartialDerivativesOf_OutputFunction_wrt_Inputs)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiBaseUnit(object):
    xmlTagName = 'BaseUnit'

    def __init__(self):
        self.kg     = None # int
        self.m      = None # int
        self.s      = None # int
        self.A      = None # int
        self.K      = None # int
        self.mol    = None # int
        self.cd     = None # int
        self.rad    = None # int
        self.factor = None # float
        self.offset = None # float

    def to_xml(self, tag):
        addAttribute(tag, 'kg',       self.kg)
        addAttribute(tag, 'm',        self.m)
        addAttribute(tag, 's',        self.s)
        addAttribute(tag, 'A',        self.A)
        addAttribute(tag, 'K',        self.K)
        addAttribute(tag, 'mol',      self.mol)
        addAttribute(tag, 'cd',       self.cd)
        addAttribute(tag, 'rad',      self.rad)
        addAttribute(tag, 'factor',   self.factor)
        addAttribute(tag, 'offset',   self.offset)
        
    @classmethod
    def from_xml(cls, tag):
        pass

class fmiUnit(object):
    xmlTagName = 'Unit'

    def __init__(self):
        self.name     = ''
        self.baseUnit = None # fmiBaseUnit

    def to_xml(self, tag):
        addAttribute(tag, 'name', self.name, required = True)
        addObject(tag, self.baseUnit)

    @classmethod
    def from_xml(cls, tag):
        pass
        
class fmiSimpleType(object):
    xmlTagName = 'SimpleType'

    def __init__(self):
        self.name        = ''   # Required: string
        self.description = ''   # string
        self.type        = None # Required: fmiReal, fmiInteger, fmiBoolean, fmiString, fmiEnumeration
    
    def to_xml(self, tag):
        addAttribute(tag, 'name',        self.name, required = True)
        addAttribute(tag, 'description', self.description)
        addObject(tag, self.type, required = True)
        
    @classmethod
    def from_xml(cls, tag):
        pass

class fmiReal(object):
    xmlTagName = 'Real'

    def __init__(self):
        self.quantity           = None # string
        self.unit               = None # string
        self.displayUnit        = None # string
        self.relativeQuantity   = None # bool
        self.min                = None # float
        self.max                = None # float
        self.nominal            = None # float
        self.unbounded          = None # bool

    def to_xml(self, tag):
        addAttribute(tag, 'quantity',         self.quantity)
        addAttribute(tag, 'unit',             self.unit)
        addAttribute(tag, 'displayUnit',      self.displayUnit)
        addAttribute(tag, 'relativeQuantity', self.relativeQuantity)
        addAttribute(tag, 'min',              self.min)
        addAttribute(tag, 'max',              self.max)
        addAttribute(tag, 'nominal',          self.nominal)
        addAttribute(tag, 'unbounded',        self.unbounded)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiInteger(object):
    xmlTagName = 'Integer'

    def __init__(self):
        self.quantity = None # string
        self.min      = None # float
        self.max      = None # float

    def to_xml(self, tag):
        addAttribute(tag, 'quantity', self.quantity)
        addAttribute(tag, 'min',      self.min)
        addAttribute(tag, 'max',      self.max)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiBoolean(object):
    xmlTagName = 'Boolean'

    def __init__(self):
        pass

    def to_xml(self, tag):
        pass

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiString(object):
    xmlTagName = 'String'

    def __init__(self):
        pass

    def to_xml(self, tag):
        pass

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiEnumeration(object):
    xmlTagName = 'Enumeration'

    def __init__(self):
        self.quantity = None # string
        self.items    = []   # array of fmiEnumerationItem

    def to_xml(self, tag):
        addAttribute(tag, 'quantity', self.quantity)
        for obj in self.items:
            addObject(tag, obj)

    @classmethod
    def from_xml(cls, tag):
        pass
        
class fmiEnumerationItem(object):
    xmlTagName = 'Item'

    def __init__(self):
        self.name        = None # string
        self.value       = None # int
        self.description = None # string

    def to_xml(self, tag):
        addAttribute(tag, 'name',        self.name)
        addAttribute(tag, 'description', self.description)
        addAttribute(tag, 'value',       self.value)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiLogCategory(object):
    xmlTagName = 'Category'

    logEvents                = 'logEvents'
    logSingularLinearSystems = 'logSingularLinearSystems'
    logNonlinearSystems      = 'logNonlinearSystems'
    logDynamicStateSelection = 'logDynamicStateSelection'

    def __init__(self):
        self.name = None # Required: enum or user-defined string

    def to_xml(self, tag):
        addAttribute(tag, 'name', self.name, required = True)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiDefaultExperiment(object):
    xmlTagName = 'DefaultExperiment'

    def __init__(self):
        self.startTime = None # float
        self.stopTime  = None # float
        self.tolerance = None # float

    def to_xml(self, tag):
        addAttribute(tag, 'startTime', self.startTime)
        addAttribute(tag, 'stopTime',  self.stopTime)
        addAttribute(tag, 'tolerance', self.tolerance)

    @classmethod
    def from_xml(cls, tag):
        pass
    
class fmiVendorAnnotation(object):
    xmlTagName = 'Tool'

    def __init__(self):
        self.name = None # string

    def to_xml(self, tag):
        addAttribute(tag, 'name', self.name, required = True)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiScalarVariable(object):
    xmlTagName = 'ScalarVariable'

    causalityParameter = 'parameter'
    causalityInput     = 'input'
    causalityOutput    = 'output'
    causalityLocal     = 'local'

    variabilityConstant   = 'constant'
    variabilityFixed      = 'fixed'
    variabilityTunable    = 'tunable'
    variabilityDiscrete   = 'discrete'
    variabilityContinuous = 'continuous'

    initialExact      = 'exact'
    initialApprox     = 'approx'
    initialCalculated = 'calculated'

    def __init__(self):
        self.name           = None # string
        self.valueReference = None # unsigned int
        self.description    = None # string
        self.causality      = None # string
        self.variability    = None # string
        self.initial        = None # string

    def to_xml(self, tag):
        addAttribute(tag, 'name',             self.name, required = True)
        addAttribute(tag, 'valueReference',   self.valueReference, required = True)
        addAttribute(tag, 'description',      self.description)
        addAttribute(tag, 'causality',        self.causality)
        addAttribute(tag, 'variability',      self.variability)
        addAttribute(tag, 'initial',          self.initial)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiModelStructure(object):
    xmlTagName = 'ModelStructure'

    def __init__(self):
        self.Inputs       = [] # array of fmiInput
        self.Derivatives  = [] # array of fmiDerivative
        self.Outputs      = [] # array of fmiOutput

    def to_xml(self, tag):
        addObjects(tag, 'Inputs',      self.Inputs)
        addObjects(tag, 'Derivatives', self.Derivatives)
        addObjects(tag, 'Outputs',     self.Outputs)

    @classmethod
    def from_xml(cls, tag):
        pass
        
class fmiInput(object):
    xmlTagName = 'Input'

    def __init__(self):
        self.name       = None # Required: string
        self.derivative = None # int

    def to_xml(self, tag):
        addAttribute(tag, 'name',       self.name, required = True)
        addAttribute(tag, 'derivative', self.derivative)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiDerivative(object):
    xmlTagName = 'Derivative'

    def __init__(self):
        self.name  = None # Required: string
        self.state = None # Required: string

    def to_xml(self, tag):
        addAttribute(tag, 'name',  self.name,  required = True)
        addAttribute(tag, 'state', self.state, required = True)

    @classmethod
    def from_xml(cls, tag):
        pass

class fmiOutput(object):
    xmlTagName = 'Output'

    def __init__(self):
        self.name       = None # Required: string
        self.derivative = None # int

    def to_xml(self, tag):
        addAttribute(tag, 'name',       self.name, required = True)
        addAttribute(tag, 'derivative', self.derivative)

    @classmethod
    def from_xml(cls, tag):
        pass

'''
class fmi(object):
    xmlTagName = ''

    def __init__(self):
        pass

    def to_xml(self, tag):
        pass

    @classmethod
    def from_xml(cls, tag):
        pass
'''
if __name__ == "__main__":
    fmi_model = fmiModelDescription()

    # Required
    fmi_model.modelName                  = 'name' #*
    fmi_model.guid                       = '{8c4e810f-3df3-4a00-8276-176fa3c9f9e0}' #*
    fmi_model.description                = ''
    fmi_model.author                     = 'Dragan Nikolic'
    fmi_model.version                    = '1.0'
    fmi_model.copyright                  = 'Dragan Nikolic'
    fmi_model.license                    = 'GNU GPL'
    fmi_model.generationTool             = 'DAE Tools'
    fmi_model.generationDateAndTime      = ''
    fmi_model.variableNamingConvention   = fmiModelDescription.variableNamingConventionStructured
    fmi_model.numberOfEventIndicators    = 0

    cos = fmiCoSimulation()
    cos.modelIdentifier                        = 'mi' #*
    cos.needsExecutionTool                     = False
    cos.canHandleVariableCommunicationStepSize = True
    cos.canHandleEvents                        = True
    cos.canInterpolateInputs                   = False
    cos.maxOutputDerivativeOrder               = 1
    cos.canRunAsynchronuously                  = False
    cos.canSignalEvents                        = True
    cos.canBeInstantiatedOnlyOncePerProcess    = False
    cos.canNotUseMemoryManagementFunctions     = True
    cos.canGetAndSetFMUstate                   = False
    cos.canSerializeFMUstate                   = False
    cos.providesPartialDerivativesOf_DerivativeFunction_wrt_States = False
    cos.providesPartialDerivativesOf_DerivativeFunction_wrt_Inputs = False
    cos.providesPartialDerivativesOf_OutputFunction_wrt_States     = False
    cos.providesPartialDerivativesOf_OutputFunction_wrt_Inputs     = False
    fmi_model.CoSimulation = cos

    ms = fmiModelStructure()
    
    i = fmiInput()
    i.name       = 'i' #*
    i.derivative = None
    ms.Inputs = [i]

    d = fmiDerivative()
    d.name  = 'd' #*
    d.state = 's' #*
    ms.Derivatives = [d]

    o = fmiOutput()
    o.name       = 'o' #*
    o.derivative = None
    ms.Outputs = [o]

    fmi_model.ModelStructure = ms

    param = fmiScalarVariable()
    param.name           = 'parameter' #*
    param.valueReference = 0 # *
    param.description    = ''
    param.causality      = fmiScalarVariable.causalityParameter
    param.variability    = fmiScalarVariable.variabilityFixed
    param.initial        = fmiScalarVariable.initialExact

    no_points_in_domain = fmiScalarVariable()
    no_points_in_domain.name           = 'no_points_in_domain' #*
    no_points_in_domain.valueReference = 0 # *
    no_points_in_domain.description    = ''
    no_points_in_domain.causality      = fmiScalarVariable.causalityLocal
    no_points_in_domain.variability    = fmiScalarVariable.variabilityConstant
    no_points_in_domain.initial        = fmiScalarVariable.initialExact

    domain_points = fmiScalarVariable()
    domain_points.name           = 'domain_points' #*
    domain_points.valueReference = 0 # *
    domain_points.description    = ''
    domain_points.causality      = fmiScalarVariable.causalityParameter
    domain_points.variability    = fmiScalarVariable.variabilityTunable
    domain_points.initial        = fmiScalarVariable.initialExact

    assigned = fmiScalarVariable()
    assigned.name           = 'assigned' #*
    assigned.valueReference = 0 # *
    assigned.description    = ''
    assigned.causality      = fmiScalarVariable.causalityParameter
    assigned.variability    = fmiScalarVariable.variabilityTunable
    assigned.initial        = fmiScalarVariable.initialExact

    algebraic = fmiScalarVariable()
    algebraic.name           = 'algebraic' #*
    algebraic.valueReference = 0 # *
    algebraic.description    = ''
    algebraic.causality      = fmiScalarVariable.causalityLocal
    algebraic.variability    = fmiScalarVariable.variabilityContinuous
    algebraic.initial        = fmiScalarVariable.initialCalculated

    diff = fmiScalarVariable()
    diff.name           = 'diff' #*
    diff.valueReference = 0 # *
    diff.description    = ''
    diff.causality      = fmiScalarVariable.causalityLocal
    diff.variability    = fmiScalarVariable.variabilityContinuous
    diff.initial        = fmiScalarVariable.initialCalculated

    stn = fmiScalarVariable()
    stn.name           = 'stn' #*
    stn.valueReference = 0 # *
    stn.description    = ''
    stn.causality      = fmiScalarVariable.causalityParameter
    stn.variability    = fmiScalarVariable.variabilityDiscrete
    stn.initial        = fmiScalarVariable.initialExact

    port_in = fmiScalarVariable()
    port_in.name           = 'port_in' #*
    port_in.valueReference = 0 # *
    port_in.description    = ''
    port_in.causality      = fmiScalarVariable.causalityInput
    port_in.variability    = fmiScalarVariable.variabilityContinuous
    port_in.initial        = fmiScalarVariable.initialExact

    port_out = fmiScalarVariable()
    port_out.name           = 'port_out' #*
    port_out.valueReference = 0 # *
    port_out.description    = ''
    port_out.causality      = fmiScalarVariable.causalityOutput
    port_out.variability    = fmiScalarVariable.variabilityContinuous
    port_out.initial        = fmiScalarVariable.initialCalculated

    fmi_model.ModelVariables = [param, no_points_in_domain, domain_points, assigned, algebraic, diff, stn, port_in, port_out]

    # Optional
    unit = fmiUnit()
    unit.name = 'unitname' # *
    bu = fmiBaseUnit()
    bu.factor = 1.2
    bu.offset = 0.1
    bu.m = -2
    unit.baseUnit = bu
    fmi_model.UnitDefinitions = [unit]

    t1 = fmiSimpleType()
    t1.name = 't1' #*
    t1.description = ''
    t = fmiReal()
    t.quantity           = None
    t.unit               = ''
    t.displayUnit        = None
    t.relativeQuantity   = True
    t.min                = -1E5
    t.max                = +1E5
    t.nominal            = 0.0
    t.unbounded          = True
    t1.type = t #*

    t2 = fmiSimpleType()
    t2.name = 't2' #*
    t2.description = None
    t = fmiInteger() #*
    t.quantity = None
    t.min      = -int(1E5)
    t.max      = +int(1E5)
    t2.type = t #*

    t3 = fmiSimpleType()
    t3.name = 't3' #*
    t3.description = None
    t = fmiString()
    t3.type = t

    t4 = fmiSimpleType()
    t4.name = 't4' #*
    t4.description = None
    t = fmiEnumeration()
    t.quantity = ''
    i1 = fmiEnumerationItem()
    i1.name        = 'i1' #*
    i1.value       = 0 #*
    i1.description = None
    i2 = fmiEnumerationItem()
    i2.name        = 'i2' #*
    i2.value       = 1 #*
    i2.description = None
    t.items = [i1, i2]
    t4.type = t #*

    fmi_model.TypeDefinitions = [t1, t2, t3, t4]

    ann = fmiVendorAnnotation()
    ann.name = 'vann' #*
    fmi_model.VendorAnnotations = [ann]

    defex = fmiDefaultExperiment()
    defex.startTime = 0.0
    defex.stopTime  = 10.0
    defex.tolerance = 1E-5
    fmi_model.DefaultExperiment = defex

    filename = 'sample_fmi.xml'
    fmi_model.to_xml(filename)

    #fmi_model_copy = fmiModelDescription.from_xml(filename)
    #filename2 = 'sample_fmi - copy.xml'
    #fmi_model_copy.to_xml(filename2)
