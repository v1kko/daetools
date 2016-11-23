#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           membrane.py
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
import sys, tempfile, math, numpy
from time import localtime, strftime
from daetools.pyDAE import *
from pyUnits import m, kg, s, K, Pa, mol, J, W
try:
    from membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                        pressure_t, length_t, diffusivity_t, area_t, gij_t, Gij_dTheta_t, J_theta_t
except Exception as e:
    from .membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                         pressure_t, length_t, diffusivity_t, area_t, gij_t, Gij_dTheta_t, J_theta_t

class Membrane(daeModel):
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
         - Dij (i, i)
         - T
         - Lenght
         - Area
         - Thickness
        """

        self.z          = daeDomain("z",  self, unit(), "Axial domain")
        self.r          = daeDomain("r",  self, unit(), "Radial domain")
        self.Nc         = daeDomain("Nc", self, unit(), "Number of components")

        self.Ro         = daeParameter("Ro",    kg/(m**3), self, "")
        self.Rc         = daeParameter("Rc",    J/(mol*K), self, "")
        self.B          = daeParameter("B",     Pa**(-1),  self, "", [self.Nc])
        self.Qsat       = daeParameter("Q_sat", mol/kg,    self, "", [self.Nc])

        self.Flux       = daeVariable("Flux",       molar_flux_t,    self, "", [self.Nc, self.z])
        self.Xinlet     = daeVariable("X_inlet",    fraction_t,      self, "", [self.Nc, self.z])
        self.Xoutlet    = daeVariable("X_outlet",   fraction_t,      self, "", [self.Nc, self.z])

        self.T          = daeVariable("T",          temperature_t,   self, "", [])
        self.Pinlet     = daeVariable("P_inlet",    pressure_t,      self, "", [self.z])
        self.Poutlet    = daeVariable("P_outlet",   pressure_t,      self, "", [self.z])

        self.Gij        = daeVariable("G_ij",       gij_t,           self, "", [self.Nc, self.Nc, self.z, self.r])
        self.Dij        = daeVariable("D_ij",       diffusivity_t,   self, "", [self.Nc, self.Nc, self.z, self.r])
        self.Di         = daeVariable("D_i",        diffusivity_t,   self, "", [self.Nc])

        self.Theta      = daeVariable("&theta;",    fraction_t,      self, "", [self.Nc, self.z, self.r])
        self.Gij_dTheta = daeVariable("Gij_dTheta", Gij_dTheta_t,    self, "", [self.Nc, self.Nc, self.z, self.r])
        self.J_theta    = daeVariable("J_theta",    J_theta_t,       self, "", [self.Nc, self.Nc, self.z, self.r])

        self.Length     = daeVariable("Length",     length_t,        self, "", [])
        self.Area       = daeVariable("Area",       area_t,          self, "", [])
        self.Thickness  = daeVariable("Thickness",  length_t,        self, "", [])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Inlet BCs
        eq = self.CreateEquation("BCinlet_Theta")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r0 = eq.DistributeOnDomain(self.r,  eLowerBound, 'r_0')
        eq.Residual = self.Theta(i, z, r0) - \
                            self.B(i) * self.Xinlet(i, z)  * self.Pinlet(z) /    \
                            (1 + Sum(self.B.array('*') * self.Xinlet.array('*', z) * self.Pinlet(z)))

        # Outlet BCs
        eq = self.CreateEquation("BCoutlet_Theta")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z  = eq.DistributeOnDomain(self.z,  eClosedClosed)
        rR = eq.DistributeOnDomain(self.r,  eUpperBound, 'r_R')
        eq.Residual = self.Theta(i, z, rR) - \
                      self.B(i) * self.Xoutlet(i, z) * self.Poutlet(z) /     \
                      (1 + Sum(self.B.array('*') * self.Xoutlet.array('*', z)  * self.Poutlet(z)))

        # Flux through the porous support
        self.stnOperatingMode = self.STN('OperatingMode')

        # Fick's law Case
        self.STATE('sFickLaw')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedOpen)
        eq.Residual = self.Flux(i, z) + \
                      self.Ro() * self.Qsat(i) * self.Di(i) * d(self.Theta(i, z, r), self.r) / self.Thickness()

        eq = self.CreateEquation("J_theta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.Residual = self.J_theta(i, j, z, r)

        eq = self.CreateEquation("Gij_dTheta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.Residual = self.Gij_dTheta(i, j, z, r)


        # General Maxwell Stefan Case
        self.STATE('sMaxwellStefan')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedOpen)
        eq.Residual = self.Flux(i, z) + \
                      (self.Qsat(i) * self.Di(i)) * Sum(self.J_theta.array(i, '*', z, r)) + \
                      (self.Qsat(i) * self.Di(i)) * self.Ro() * Sum(self.Gij_dTheta.array(i, '*', z, r))

        eq = self.CreateEquation("J_theta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        cond = (i() - j())/(i() - j() + 1E-15)
        eq.Residual = self.J_theta(i, j, z, r) - cond * ( \
                         self.Flux(i, z) * self.Theta(j, z, r) / (self.Qsat(i) * self.Dij(i, j, z, r)) - \
                         self.Flux(j, z) * self.Theta(i, z, r) / (self.Qsat(j) * self.Dij(i, j, z, r)) \
                      )

        eq = self.CreateEquation("Gij_dTheta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.Residual = self.Gij_dTheta(i, j, z, r) - \
                      self.Gij(i, j, z, r) * d(self.Theta(j, z, r), self.r) / self.Thickness()

        # "Single file" diffusion Case
        # Friction between molecules less important than friction with the wall: Dij much larger than Di
        self.STATE('sMaxwellStefan_Dijoo')
        eq = self.CreateEquation("Flux")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedOpen)
        eq.Residual = self.Flux(i, z) + \
                      self.Ro() * self.Qsat(i) * self.Di(i) * Sum(self.Gij_dTheta.array(i, '*', z, r))

        eq = self.CreateEquation("J_theta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.Residual = self.J_theta(i, j, z, r)

        eq = self.CreateEquation("Gij_dTheta")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        j = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'j')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.Residual = self.Gij_dTheta(i, j, z, r) - \
                      self.Gij(i, j, z, r) * d(self.Theta(j, z, r), self.r) / self.Thickness()
        
        self.END_STN()

      
        eq = self.CreateEquation("GammaFactor")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        k = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'k')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        # Condition expression:
        # if i == k:
        #   expr = 1, because 1 - (2-2)/(2-2+eps) = 1 - 0/0 = 1
        # else:
        #   expr = 0, because 1 - (2-3)/(2-3+eps) = 1 - (-1)/(-1) = 1 - 1 = 0
        cond = 1 - (i() - k())/(i() - k() + 1E-15)
        eq.Residual = (self.Gij(i, k, z, r) - cond) * (1 - Sum(self.Theta.array('*', z, r))) - self.Theta(i, z, r)

        eq = self.CreateEquation("Dij")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        k = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'k')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        r = eq.DistributeOnDomain(self.r,  eClosedClosed)
        eq.CheckUnitsConsistency = False
        eq.Residual = self.Dij(i, k, z, r) - (self.Di(i) ** ((self.Theta(i, z, r) / (self.Theta(i, z, r) + self.Theta(k, z, r) + 1e-10)))) * \
                                             (self.Di(k) ** ((self.Theta(k, z, r) / (self.Theta(i, z, r) + self.Theta(k, z, r) + 1e-10))))
