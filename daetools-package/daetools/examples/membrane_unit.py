#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           membrane_unit.py
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
__doc__ = """
"""
import sys, tempfile, math, numpy
from time import localtime, strftime
from daetools.pyDAE import *
from pyUnits import m, kg, s, K, Pa, mol, J, W
try:
    from .membrane_variable_types import molar_flux_t, molar_concentration_t, fraction_t, temperature_t, recovery_t, \
                                         pressure_t, length_t, diffusivity_t, area_t, volume_flowrate_t, selectivity_t
    from .compartment import Compartment
    from .support import Support
    from .membrane import Membrane
except Exception as e:
    from membrane_variable_types import molar_flux_t, molar_concentration_t, fraction_t, temperature_t, recovery_t, \
                                        pressure_t, length_t, diffusivity_t, area_t, volume_flowrate_t, selectivity_t
    from compartment import Compartment
    from support import Support
    from membrane import Membrane

class MembraneUnit(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.F = Compartment("Feed", self)
        self.M = Membrane("Membrane", self)
        self.S = Support("Support", self)
        self.P = Compartment("Permeate", self)

        self.Nc = daeDomain("Nc", self, unit(), "Number of components")
        self.z  = daeDomain("z",  self, unit(), "Axial domain")

        self.Tref  = daeParameter("T_ref",  K,  self, "")
        self.Pref  = daeParameter("P_ref",  Pa, self, "")
        self.Tfeed = daeParameter("T_feed", K,  self, "Feed temperature")

        self.Purity_feed        = daeVariable("Purity_feed",        fraction_t,        self, "", [self.Nc])
        self.Purity_permeate    = daeVariable("Purity_permeate",    fraction_t,        self, "", [self.Nc])
        self.Recovery_feed      = daeVariable("Recovery_feed",      recovery_t,        self, "", [self.Nc])
        self.Selectivity        = daeVariable("Selectivity",        selectivity_t,     self, "", [self.Nc, self.Nc, self.z])

        self.Phigh              = daeVariable("P_high",             pressure_t,        self, "", [])
        self.Plow               = daeVariable("P_low",              pressure_t,        self, "", [])

        self.MembraneArea       = daeVariable("MembraneArea",       area_t,            self, "", [])
        self.MembraneThickness  = daeVariable("MembraneThickness",  length_t,          self, "", [])
        self.SupportThickness   = daeVariable("SupportThickness",   length_t,          self, "", [])

        self.Qfeed_stp          = daeVariable("Qfeed_stp",          volume_flowrate_t, self, "", [])
        self.Qsweep_stp         = daeVariable("Qsweep_stp",         volume_flowrate_t, self, "", [])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("Recovery_feed")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        eq.Residual = self.Recovery_feed(i) * (self.F.Cin(i) * self.F.Qin()) - self.F.Qout() * self.F.Cout(i)

        eq = self.CreateEquation("Purity_feed")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        eq.Residual = self.Purity_feed(i) - self.F.Xout(i)

        eq = self.CreateEquation("Purity_permeate")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        eq.Residual = self.Purity_permeate(i) - self.P.Xout(i)

        eq = self.CreateEquation("Selectivity")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        eq.Residual = self.Selectivity(i,j,z) * (self.F.X(i,z) * self.P.X(j,z)) - (self.P.X(i,z) * self.F.X(j,z))

        # Fluxes at the Feed-Membrane, Membrane-Support,
        # Support-Retentate compartments are equal
        eq = self.CreateEquation("Feed_Flux")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.F.Flux(i, z) - self.M.Flux(i, z)

        eq = self.CreateEquation("Support_Flux")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.S.Flux(i, z) - self.M.Flux(i, z)

        eq = self.CreateEquation("Permeate_Flux")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.P.Flux(i, z) + self.S.Flux(i, z)

        # Gas mole fraction at the Feed-Membrane, and Membrane-Support,
        # Support-Retentate compartments
        eq = self.CreateEquation("Membrane_Xinlet")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.F.X(i, z) - self.M.Xinlet(i, z)

        eq = self.CreateEquation("Support_Xinlet")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.S.Xinlet(i, z) - self.M.Xoutlet(i, z)

        eq = self.CreateEquation("Permeate_X")
        i = eq.DistributeOnDomain(self.F.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.P.X(i, z) - self.S.Xoutlet(i, z)

        # Pressures at the Feed-Membrane, and Membrane-Support,
        # Support-Retentate compartments are equal
        eq = self.CreateEquation("Membrane_Pinlet")
        z = eq.DistributeOnDomain(self.F.z,  eClosedClosed)
        eq.Residual = self.F.P(z) - self.M.Pinlet(z)

        eq = self.CreateEquation("Support_Pinlet")
        z = eq.DistributeOnDomain(self.S.z,  eClosedClosed)
        eq.Residual = self.S.Pinlet(z) - self.M.Poutlet(z)

        eq = self.CreateEquation("Support_Poutlet")
        z = eq.DistributeOnDomain(self.P.z,  eClosedClosed)
        eq.Residual = self.P.P(z) - self.S.Poutlet(z)

        # Temperatures at the Feed-Membrane, and Membrane-Support,
        # Support-Retentate compartments are equal
        eq = self.CreateEquation("Feed_T")
        eq.Residual = self.F.T() - self.Tfeed()

        eq = self.CreateEquation("Membrane_T")
        eq.Residual = self.M.T() - self.Tfeed()

        eq = self.CreateEquation("Support_S")
        eq.Residual = self.S.T() - self.Tfeed()

        eq = self.CreateEquation("Permeate_T")
        eq.Residual = self.P.T() - self.Tfeed()

