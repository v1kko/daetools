#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
************************************************************************************
                           support.py
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
************************************************************************************
"""
__doc__ = """
"""

import sys, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from pyUnits import m, kg, s, K, Pa, mol, J, W
try:
    from membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                        pressure_t, length_t, diffusivity_t, area_t, gij_t, Gij_dTheta_t, J_theta_t
except Exception as e:
    from .membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                         pressure_t, length_t, diffusivity_t, area_t, gij_t, Gij_dTheta_t, J_theta_t

class Support(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        """
        The model calculate:
          - Xoutlet (z)
          - Poutlet (z)
          - X (i, z, r)
          - P(z, r)

        For input:
         - Parameters (e, MW)
         - Flux (i, z)
         - Xinlet (i, z)
         - Pinlet (z)
         - Di (i)
         - T
         - Thickness
        """

        self.z    = daeDomain("z",  self, unit(), "Axial domain")
        self.r    = daeDomain("r",  self, unit(), "Radial domain")
        self.Nc   = daeDomain("Nc", self, unit(), "Number of components")

        self.e    = daeParameter("e",    unit(),    self, "")
        self.Rc   = daeParameter("Rc",   J/(mol*K), self, "")

        self.Flux       = daeVariable("Flux",       molar_flux_t,    self, "", [self.Nc, self.z])
        self.Xinlet     = daeVariable("X_inlet",    fraction_t,      self, "", [self.Nc, self.z])
        self.Xoutlet    = daeVariable("X_outlet",   fraction_t,      self, "", [self.Nc, self.z]) #[*]

        self.T          = daeVariable("T",          temperature_t,   self, "", [])
        self.P          = daeVariable("P",          pressure_t,      self, "", [self.z, self.r]) #[*]
        self.Pinlet     = daeVariable("P_inlet",    pressure_t,      self, "", [self.z])
        self.Poutlet    = daeVariable("P_outlet",   pressure_t,      self, "", [self.z]) #[*]

        self.X          = daeVariable("X",          fraction_t,      self, "", [self.Nc, self.z, self.r]) #[*]
        self.Di         = daeVariable("D_i",        diffusivity_t,   self, "", [self.Nc])
        self.Dij        = daeVariable("D_ij",       diffusivity_t,   self, "", [self.Nc, self.Nc])

        self.Thickness  = daeVariable("Thickness",  length_t,        self, "", [])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Inlet BCs
        eq = self.CreateEquation("BCinlet_X")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r0 = eq.DistributeOnDomain(self.r,  eLowerBound, 'r_0')
        eq.Residual = self.X(i, z, r0) - self.Xinlet(i, z)

        eq = self.CreateEquation("BCoutlet_X")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        rR = eq.DistributeOnDomain(self.r,  eUpperBound, 'r_R')
        eq.Residual = self.X(i, z, rR) - self.Xoutlet(i, z)

        eq = self.CreateEquation("BCinlet_P")
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r0 = eq.DistributeOnDomain(self.r,  eLowerBound, 'r_0')
        eq.Residual = self.Pinlet(z) - self.P(z, r0)

        eq = self.CreateEquation("BCoutlet_P")
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        rR = eq.DistributeOnDomain(self.r,  eUpperBound, 'r_R')
        eq.Residual = self.Poutlet(z) - self.P(z, rR)

        # Flux through the porous support
        self.stnOperatingMode = self.STN('OperatingMode')

        self.STATE('sNoResistance')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eOpenClosed)
        eq.Residual = d(self.X(i, z, r), self.r)

        self.STATE('sFickLaw')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eOpenClosed)
        eq.Residual = self.Flux(i, z) + \
                      self.e() * self.Dij(i, i) * d(self.X(i, z, r), self.r) * self.P(z, r) / (self.Rc() * self.T() * self.Thickness())

        self.STATE('sMaxwellStefan')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eOpenClosed)
        eq.Residual = - d(self.X(i, z, r) * self.P(z, r), self.r) / (self.Rc() * self.T() * self.Thickness()) - \
                        Sum( (self.Flux(i, z) * self.X.array('*', z, r) - self.Flux.array('*', z) * self.X(i, z, r)) / (self.e() * self.Dij.array(i, '*')) ) - \
                         ( 1 - Sum(self.X.array('*', z, r)) ) * self.Flux(i, z) / (self.e() * self.Di(i))

        self.END_STN()

        eq = self.CreateEquation("P")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        r = eq.DistributeOnDomain(self.r, eClosedOpen)
        eq.Residual = d(self.P(z, r), self.r)
