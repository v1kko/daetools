#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
************************************************************************************
                           compartment.py
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
import sys, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from pyUnits import m, kg, s, K, Pa, mol, J, W, ml
try:
    from membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                        pressure_t, length_t, diffusivity_t, area_t, specific_area_t, volume_flowrate_t
except Exception as e:
    from .membrane_variable_types import velocity_t, molar_flux_t, molar_concentration_t, fraction_t, temperature_t, \
                                         pressure_t, length_t, diffusivity_t, area_t, specific_area_t, volume_flowrate_t

class Compartment(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.z          = daeDomain("z",  self, unit(), "Axial domain")
        self.Nc         = daeDomain("Nc", self, unit(), "Number of components")

        self.Rc         = daeParameter("Rc",   J/(mol*K), self, "")

        self.U          = daeVariable("U",          velocity_t,             self, "", [self.z])
        self.Flux       = daeVariable("Flux",       molar_flux_t,           self, "", [self.Nc, self.z])
        self.C          = daeVariable("C",          molar_concentration_t,  self, "", [self.Nc, self.z])
        self.X          = daeVariable("X",          fraction_t,             self, "", [self.Nc, self.z])
        self.T          = daeVariable("T",          temperature_t,          self, "", [])
        self.P          = daeVariable("P",          pressure_t,             self, "", [self.z])
        self.Length     = daeVariable("Length",     length_t,               self, "", [])
        self.Dz         = daeVariable("D_z",        diffusivity_t,          self, "", [self.Nc, self.z])
        self.aV         = daeVariable("a_V",        specific_area_t,        self, "", [])
        self.Across     = daeVariable("A_cross",    area_t,                 self, "", [])
        self.Area       = daeVariable("Area",       area_t,                 self, "", [])
        self.Xin        = daeVariable("X_in",       fraction_t,             self, "", [self.Nc])
        self.Xout       = daeVariable("X_out",      fraction_t,             self, "", [self.Nc])
        self.Qin        = daeVariable("Q_in",       volume_flowrate_t,      self, "", [])
        self.Qout       = daeVariable("Q_out",      volume_flowrate_t,      self, "", [])
        self.Cin        = daeVariable("C_in",       molar_concentration_t,  self, "", [self.Nc])
        self.Cout       = daeVariable("C_out",      molar_concentration_t,  self, "", [self.Nc])
        self.Pin        = daeVariable("P_in",       pressure_t,             self, "", [])
        self.Pout       = daeVariable("P_out",      pressure_t,             self, "", [])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Inlet BCs
        eq = self.CreateEquation("BCinlet_C")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z0 = eq.DistributeOnDomain(self.z,  eLowerBound, 'z_0')
        eq.Residual = self.U(z0) * (self.C(i, z0) - self.Cin(i)) - \
                      self.Dz(i, z0) * d(self.C(i,z0), self.z) / self.Length()

        eq = self.CreateEquation("BCinlet_U")
        z0 = eq.DistributeOnDomain(self.z,  eLowerBound, 'z_0')
        eq.Residual = self.U(z0) - self.Qin() / self.Across()

        eq = self.CreateEquation("BCinlet_Xin")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        eq.Residual = self.Xin(i) * self.Pin() / (self.Rc() * self.T()) - self.Cin(i)

        eq = self.CreateEquation("BCinlet_Pin")
        z0 = eq.DistributeOnDomain(self.z,  eLowerBound, 'z_0')
        eq.Residual = self.P(z0) - self.Pin()
        
        # Outlet BCs
        eq = self.CreateEquation("BCoutlet_C")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        zL = eq.DistributeOnDomain(self.z,  eUpperBound)
        eq.Residual = d(self.C(i,zL), self.z)

        eq = self.CreateEquation("BCoutlet_U")
        zL = eq.DistributeOnDomain(self.z,  eUpperBound, 'z_L')
        eq.Residual = d(self.U(zL), self.z)

        eq = self.CreateEquation("BCoutlet_Xout")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        zL = eq.DistributeOnDomain(self.z,  eUpperBound, 'z_L')
        eq.Residual = self.Xout(i) * self.P(zL) / (self.Rc() * self.T()) - self.C(i, zL)

        eq = self.CreateEquation("BCoutlet_Qout")
        zL = eq.DistributeOnDomain(self.z,  eUpperBound, 'z_L')
        eq.Residual = self.U(zL) - self.Qout() / self.Across()

        eq = self.CreateEquation("BCoutlet_Cout")
        i  = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        zL = eq.DistributeOnDomain(self.z,  eUpperBound, 'z_L')
        eq.Residual = self.Cout(i) - self.C(i, zL)

        eq = self.CreateEquation("BCoutlet_Pout")
        zL = eq.DistributeOnDomain(self.z,  eUpperBound, 'z_L')
        eq.Residual = self.Pout() - self.P(zL)

        # Equations:
        eq = self.CreateEquation("P")
        z = eq.DistributeOnDomain(self.z, eOpenClosed)
        eq.Residual = self.P(z) - self.Pin()
        
        eq = self.CreateEquation("aV")
        eq.Residual = self.aV() - self.Area() / (self.Length() * self.Across())

        eq = self.CreateEquation("C")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = -self.Dz(i, z) * d2(self.C(i,z), self.z) / (self.Length() ** 2) + \
                      d(self.U(z) * self.C(i, z), self.z) / self.Length() +  \
                      self.aV() * self.Flux(i, z)


        eq = self.CreateEquation("X")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed, 'i')
        z = eq.DistributeOnDomain(self.z,  eClosedClosed)
        eq.Residual = self.X(i, z) * Sum(self.C.array('*', z)) - self.C(i, z)
        
        eq = self.CreateEquation("PVT")
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = self.P(z) / (self.Rc() * self.T()) - Sum(self.C.array('*', z))
