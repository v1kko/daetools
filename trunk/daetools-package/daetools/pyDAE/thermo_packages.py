"""********************************************************************************
                            thermo_packages.py
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
from daetools.pyDAE import *
from pyUnits import unit, m, g, kg, s, K, Pa, mol, J, W

'''###################################################################
          DAE Tools Thermo Package auxiliary wrapper class
###################################################################'''
class daeThermoPackage(daeThermoPhysicalPropertyPackage):
    def __init__(self, Name, Parent = None, Description = ""):
        daeThermoPhysicalPropertyPackage.__init__(self, Name, Parent, Description)

        self.defaultBasis = eMole
        self.defaultPhase = ''

    def LoadCapeOpen(self, packageManager, packageName, compoundIDs, compoundCASNumbers, availablePhases, defaultBasis, options):
        self.Load_CapeOpen_TPP(packageManager, packageName, compoundIDs, compoundCASNumbers, availablePhases, defaultBasis, options)
        self.defaultBasis = defaultBasis
        # If there is only one phase use it as a default.
        # Otherwise, it must be specified as a keyword argument to the property calculation functions.
        phases = list(availablePhases.items())
        if len(phases) == 1:
            self.defaultPhase = phases[0][0]
        else:
            self.defaultPhase = None

    def LoadCoolProp(self, compoundIDs, compoundCASNumbers, availablePhases, defaultBasis, options):
        self.Load_CoolProp_TPP(compoundIDs, compoundCASNumbers, availablePhases, defaultBasis, options)
        self.defaultBasis = defaultBasis
        # If there is only one phase use it as a default.
        # Otherwise, it must be specified as a keyword argument to the property calculation functions.
        phases = list(availablePhases.items())
        if len(phases) == 1:
            self.defaultPhase = phases[0][0]
        else:
            self.defaultPhase = None

    # Auxiliary functions
    def SetDefaultBasis(self, defaultBasis):
        self.defaultBasis = defaultBasis

    def SetDefaultPhase(defaultPhase):
        self.defaultPhase = defaultPhase

    # Transport properties:
    def cp(self, P, T, x, **kwargs):
        # Specific heat capacity, J/(kg*K) for mass basis or J/(mol*K) for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("heatCapacityCp", P, T, x, phase, basis)

    def kappa(self, P, T, x, **kwargs):
        # Thermal conductivity, W/(m*K)
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("thermalConductivity", P, T, x, phase, eUndefinedBasis)

    def mu(self, P, T, x, **kwargs):
        # Viscosity, Pa*s
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("viscosity", P, T, x, phase, eUndefinedBasis)

    def Dab(self, P, T, x, **kwargs):
        # Binary diffusion coefficient, m**2/s
        # Vector property (tensor rank 2)
        # Access items using Dab[i,j] = Dab(i*Nc + j)
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("diffusionCoefficient", P, T, x, phase, eUndefinedBasis)

    # Thermodynamic properties:
    def MW(self, P, T, x, **kwargs):
        # Molecular weight, dimensionless
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("molecularWeight", P, T, x, phase, eUndefinedBasis)

    def M(self, P, T, x, **kwargs):
        # Molar mass, kg/mol
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        # TPP package returns dimensionless MW (relative molar mass).
        # Multiply it by the 'molar mass constant' (1 g/mol) and convert it to kg/mol.
        MW = self.CalcSinglePhaseScalarProperty("molecularWeight", P, T, x, phase, eUndefinedBasis)
        return MW * Constant(0.001 * kg/mol)

    def rho(self, P, T, x, **kwargs):
        # Density, kg/m**3 for mass basis or mol/m**3 for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("density", P, T, x, phase, basis)

    def h(self, P, T, x, **kwargs):
        # Enthalpy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("enthalpy", P, T, x, phase, basis)

    def s(self, P, T, x, **kwargs):
        # Entropy, J/(kg*K) for mass basis or J/(mol*K) for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("entropy", P, T, x, phase, basis)

    def G(self, P, T, x, **kwargs):
        # Enthalpy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("gibbsEnergy", P, T, x, phase, basis)

    def H(self, P, T, x, **kwargs):
        # Enthalpy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("helmholtzEnergy", P, T, x, phase, basis)

    def I(self, P, T, x, **kwargs):
        # Internal energy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("internalEnergy", P, T, x, phase, basis)


    def h_E(self, P, T, x, **kwargs):
        # Excess enthalpy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessEnthalpy", P, T, x, phase, basis)

    def s_E(self, P, T, x, **kwargs):
        # Excess entropy, J/(kg*K) for mass basis or J/(mol*K) for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessEntropy", P, T, x, phase, basis)

    def G_E(self, P, T, x, **kwargs):
        # Excess Gibbs energy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessGibbsEnergy", P, T, x, phase, basis)

    def H_E(self, P, T, x, **kwargs):
        # Excess Helmholtz energy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessHelmholtzEnergy", P, T, x, phase, basis)

    def I_E(self, P, T, x, **kwargs):
        # Excess internal energy, J/kg for mass basis or J/mol for molar basis
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessInternalEnergy", P, T, x, phase, basis)

    def V_E(self, P, T, x, **kwargs):
        # Excess volume, m**3
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseScalarProperty("excessVolume", P, T, x, phase, basis)


    def f(self, P, T, x, **kwargs):
        # Fugacity, Pa
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("fugacity",  P, T, x, phase, eUndefinedBasis)

    def log_f(self, P, T, x, **kwargs):
        # Natural logarithm of the fugacity (expressed in Pa), dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("logFugacity",  P, T, x, phase, eUndefinedBasis)

    def phi(self, P, T, x, **kwargs):
        # Fugacity coefficient, dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("fugacityCoefficient",  P, T, x, phase, eUndefinedBasis)

    def log_phi(self, P, T, x, **kwargs):
        # Natural logarithm of the fugacity coefficient, dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("logFugacityCofficient",  P, T, x, phase, eUndefinedBasis)

    def a(self, P, T, x, **kwargs):
        # Activity, dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("activity",  P, T, x, phase, eUndefinedBasis)

    def gamma(self, P, T, x, **kwargs):
        # Activity coefficient, dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("activityCoefficient",  P, T, x, phase, eUndefinedBasis)

    def z(self, P, T, x, **kwargs):
        # Compressibility factor, dimensionless
        # Vector property
        phase = (kwargs['phase'] if 'phase' in kwargs else self.defaultPhase)
        basis = (kwargs['basis'] if 'basis' in kwargs else self.defaultBasis)
        return self.CalcSinglePhaseVectorProperty("compressibilityFactor",  P, T, x, phase, eUndefinedBasis)

    # Two phase properties:
    def K(self, P1, T1, x1, phase1, P2, T2, x2, phase2, **kwargs):
        # K-value (ratio of fugacity coefficients), dimensionless
        # Vector property
        return self.CalcTwoPhaseVectorProperty("kvalue", P1, T1, x1, phase1, P2, T2, x2, phase2, eUndefinedBasis)

    def logK(self, P1, T1, x1, phase1, P2, T2, x2, phase2, **kwargs):
        # Natural logarithm of the K-value
        # Vector property
        return self.CalcTwoPhaseVectorProperty("logKvalue", P1, T1, x1, phase1, P2, T2, x2, phase2, eUndefinedBasis)

    def surfaceTension(self, P1, T1, x1, phase1, P2, T2, x2, phase2, **kwargs):
        # Surface tension, N/m
        return self.CalcTwoPhaseVectorProperty("surfaceTension", P1, T1, x1, phase1, P2, T2, x2, phase2, eUndefinedBasis)

