#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                          tutorial_che_6.py
                Copyright (C) Raymond B. Smith, 2016
***********************************************************************************
This program is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
from daetools.pyDAE import *
import numpy as np

from pyUnits import m, s, K, mol, J, A, V, S

__doc__ = """
Model of a lithium-ion battery based on porous electrode theory as developed
by John Newman and coworkers. In particular, the equations here are based on a summary
of the methodology by Karen E. Thomas, John Newman, and Robert M. Darling,

Thomas K., Newman J., Darling R. (2002). Mathematical Modeling of Lithium Batteries
in Advances in Lithium-ion Batteries. Springer US. 345-392.
`doi:10.1007/0-306-47508-1_13 <http://dx.doi.org/10.1007/0-306-47508-1_13>`_

A few simplifications have been made rather than implementing the more complete model described there.
For example, the following assumptions have (currently) been made:

- two porous electrodes are used rather than providing the option for a "half cell" in which
  one electrode is lithium foil.
- conductivity in the electron-conducting phase is infinite
- constant exchange current density in Butler-Volmer reaction expression
- no electrolyte convection
- constant and uniform solvent concentration (ions vary according to concentrated solution theory)
- monodisperse particles in electrode
- no volume occupied by binder, filler, etc. in the electrode

"""

# Define some variable types
conc_t = daeVariableType(
    name="conc_t", units=mol/(m**3), lowerBound=0,
    upperBound=1e20, initialGuess=1.00, absTolerance=1e-6)
elec_pot_t = daeVariableType(
    name="elec_pot_t", units=V, lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
current_dens_t = daeVariableType(
    name="current_dens_t", units=A/m**2, lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
rxn_t = daeVariableType(
    name="rxn_t", units=mol/(m**2 * s), lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)

def kappa(c):
    """Return the conductivity of the electrolyte in S/m as a function of concentration in M."""
    out = 0.1  # S/m
    return out * Constant(1 * S/m)

def D(c):
    """Return electrolyte diffusivity (in m^2/s) as a function of concentration in M."""
    out = 1e-10  # m**2/s
    return out * Constant(1 * m**2/s)

def thermodynamic_factor(c):
    """Return the electrolyte thermodynamic factor as a function of concentration in M."""
    out = 1
    return out

def t_p(c):
    """Return the electrolyte cation transference number as a function of concentration in M."""
    out = 0.3 * c/c
    return out

def Ds_n(y):
    """Return diffusivity (in m^2/s) as a function of solid filling fraction, y."""
    out = 5e-9 * y/y  # m**2/s
    return out * Constant(1 * m**2/s)

def Ds_p(y):
    """Return diffusivity (in m^2/s) as a function of solid filling fraction, y."""
    out = 1e-9 * y/y  # m**2/s
    return out * Constant(1 * m**2/s)

def U_n(y):
    """Return the equilibrium potential (V vs Li) of the negative electrode active material
    as a function of solid filling fraction, y.
    """
    material = "coke"
    if material == "coke":
        # Carbon (coke) -- Fuller, Doyle, Newman, J. Electrochem. Soc., 1994
        out = -0.132 + 1.42*np.exp(-2.52*(0.5*y))
    elif material == "Li metal":
        # Lithium metal
        out = 0.
    units = Constant(1 * V) if isinstance(y, adouble) else V
    return out * units

def U_p(y):
    """Return the equilibrium potential (V vs Li) of the positive electrode active material
    as a function of solid filling fraction, y.
    """
    material = "Mn2O4"
    if material == "Mn2O4":
        # Mn2O4 -- Fuller, Doyle, Newman, J. Electrochem. Soc., 1994
        out = (4.06279 + 0.0677504*np.tanh(-21.8502*y + 12.8268) -
               0.105734*(1/((1.00167 - y)**(0.379571)) - 1.575994) -
               0.045*np.exp(-71.69*y**8) +
               0.01*np.exp(-200*(y - 0.19)))
    elif material == "Li metal":
        # Lithium metal
        out = 0.
    units = Constant(1 * V) if isinstance(y, adouble) else V
    return out * units

class ModParticle(daeModel):
    def __init__(self, Name, pindx1, pindx2, c2, y_avg, phi2, phi1, Ds, U, Parent=None, Description=""):
        daeModel.__init__(self, Name, Parent, Description)
        self.Ds = Ds
        self.U = U

        # Domain where variables are distributed
        self.r = daeDomain("r", self, m, "radial domain in particle")

        # Variables
        self.c = daeVariable("c", conc_t, self, "Concentration in the solid")
        self.c.DistributeOnDomain(self.r)
        self.j_p = daeVariable("j_p", rxn_t, self, "Rate of reaction into the solid")

        # Parameter
        self.j_0 = daeParameter("j_0", mol/(m**2 * s), self, "Exchange current density / F")
        self.alpha = daeParameter("alpha", unit(), self, "Reaction symmetry factor")
        self.c_ref = daeParameter("c_ref", mol/m**3, self, "Max conc of species in the solid")
        self.V_thermal = daeParameter("V_thermal", V, self, "Thermal voltage")
        self.R = daeParameter("R", m, self, "Radius of particle")

        self.pindx1 = pindx1
        self.pindx2 = pindx2
        self.phi2 = phi2
        self.c2 = c2
        self.phi1 = phi1
        self.y_avg = y_avg

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Mass conservation in the solid particles governed by (possibly non-linear) Ficks Law diffusion
        # Thomas et al., Eq 17
        eq = self.CreateEquation("mass_cons")
        r = eq.DistributeOnDomain(self.r, eOpenOpen)
        c = self.c(r)
        dc = d(c, self.r, eCFDM)
        D = self.Ds(c/self.c_ref())
        eq.Residual = dt(c) - 1/r()**2*d(r()**2*D*dc, self.r, eCFDM)

        # Symmetry at the center from particles with spherical geometry and symmetry
        # Thomas et al., Eq 18
        eq = self.CreateEquation("CenterSymmetry", "dc/dr = 0 at r=0")
        r = eq.DistributeOnDomain(self.r, eLowerBound)
        c = self.c(r)
        eq.Residual = d(c, self.r, eCFDM)

        # Flux at the particle surface given by the electrochemical reaction rate of (di)intercalation
        # Thomas et al., Eq 18
        eq = self.CreateEquation("SurfaceGradient", "D_s*dc/dr = j_+ at r=R_p")
        r = eq.DistributeOnDomain(self.r, eUpperBound)
        c = self.c(r)
        eq.Residual = self.Ds(c/self.c_ref()) * d(c, self.r, eCFDM) - self.j_p()

        # The rate of electrochemical reaction calculated via the Butler-Volmer equation
        # Here, we use a constant exchange current density, but other forms depending on
        # solid and electrolyte concentrations are commonly used.
        # Thomas et al., Eq 19 and 27
        c_surf = self.c(self.r.NumberOfPoints - 1)
        eta = self.phi1(self.pindx1) - self.phi2(self.pindx2) - self.U(c_surf/self.c_ref())
        eta_ndim = eta / self.V_thermal()
        # At time t=0, to aid in initialization, we use a linearized form of the reaction equation
        self.IF(Time() == Constant(0*s))
        eq = self.CreateEquation("SurfaceRxn", "Reaction rate")
        eq.Residual = self.j_p() + self.j_0() * eta_ndim
        # For the rest of the simulation, we use the full Butler-Volmer equation
        self.ELSE()
        eq = self.CreateEquation("SurfaceRxn", "Reaction rate")
        eq.Residual = self.j_p() - self.j_0() * (np.exp(-self.alpha()*eta_ndim) - np.exp((1 - self.alpha())*eta_ndim))
        self.END_IF()

        # For convenience, we also keep track of the average filling fraction in this particle.
        # This is obtained from integrating the conservation equation over the spherical
        # particle and applying the divergence theorem.
        eq = self.CreateEquation("y_avg")
        eq.Residual = self.y_avg.dt(self.pindx2) - 3*self.j_p()/(self.c_ref()*self.R())

class ModCell(daeModel):
    def __init__(self, Name, Parent=None, Description="", process_info=None):
        daeModel.__init__(self, Name, Parent, Description)
        self.process_info = process_info

        # Domains where variables are distributed
        self.x_centers_n = daeDomain("x_centers_n", self, m, "X cell-centers domain in negative electrode")
        self.x_centers_p = daeDomain("x_centers_p", self, m, "X cell-centers domain in positive electrode")
        self.x_centers_full = daeDomain("x_centers_full", self, m, "X cell-centers domain over full cell")
        self.x_faces_full = daeDomain("x_faces_full", self, m, "X cell-faces domain over full cell")

        # Variables
        # Concentration/potential in different regions of electrolyte and electrode
        self.c = daeVariable("c", conc_t, self, "Concentration in the elyte")
        self.phi2 = daeVariable("phi2", elec_pot_t, self, "Electric potential in the elyte")
        self.i2 = daeVariable("i2", current_dens_t, self, "Electrolyte current density")
        self.c.DistributeOnDomain(self.x_centers_full)
        self.phi2.DistributeOnDomain(self.x_centers_full)
        self.i2.DistributeOnDomain(self.x_faces_full)
        self.phi1_n = daeVariable("phi1_n", elec_pot_t, self, "Electric potential in bulk sld, negative")
        self.phi1_p = daeVariable("phi1_p", elec_pot_t, self, "Electric potential in bulk sld, positive")
        self.phi1_n.DistributeOnDomain(self.x_centers_n)
        self.phi1_p.DistributeOnDomain(self.x_centers_p)
        self.phiCC_n = daeVariable("phiCC_n", elec_pot_t, self, "phi at negative current collector")
        self.phiCC_p = daeVariable("phiCC_p", elec_pot_t, self, "phi at positive current collector")
        self.y_avg = daeVariable("y_avg", no_t, self, "Average filling fraction in the solid")
        self.y_avg.DistributeOnDomain(self.x_centers_full)
        self.V = daeVariable("V", elec_pot_t, self, "Applied voltage")
        self.current = daeVariable("current", current_dens_t, self, "Total current of the cell")

        # Parameters
        self.F = daeParameter("F", A*s/mol, self, "Faraday's constant")
        self.R = daeParameter("R", J/(mol*K), self, "Gas constant")
        self.T = daeParameter("T", K, self, "Temperature")
        self.a_n = daeParameter("a_n", m**(-1), self, "Reacting area per electrode volume, negative electrode")
        self.a_p = daeParameter("a_p", m**(-1), self, "Reacting area per electrode volume, positive electrode")
        self.L_n = daeParameter("L_n", m, self, "Length of negative electrode")
        self.L_s = daeParameter("L_s", m, self, "Length of separator")
        self.L_p = daeParameter("L_p", m, self, "Length of positive electrode")
        self.BruggExp_n = daeParameter("BruggExp_n", unit(), self, "Bruggeman exponent in x_n")
        self.BruggExp_s = daeParameter("BruggExp_s", unit(), self, "Bruggeman exponent in x_s")
        self.BruggExp_p = daeParameter("BruggExp_p", unit(), self, "Bruggeman exponent in x_p")
        self.poros_n = daeParameter("poros_n", unit(), self, "porosity in x_n")
        self.poros_s = daeParameter("poros_s", unit(), self, "porosity in x_s")
        self.poros_p = daeParameter("poros_p", unit(), self, "porosity in x_p")
        self.c_ref = daeParameter("c_ref", mol/m**3, self, "Reference electrolyte concentration")
        self.currset = daeParameter("currset", A/m**2, self, "current per electrode area")
        self.Vset = daeParameter("Vset", V, self, "applied voltage set point")
        self.tau_ramp = daeParameter("tau_ramp", s, self, "Time scale for ramping voltage or current")
        self.xval_cells = daeParameter("xval_cells", m, self, "coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "coordinate of cell faces")
        self.xval_cells.DistributeOnDomain(self.x_centers_full)
        self.xval_faces.DistributeOnDomain(self.x_faces_full)

        # Sub-models
        N_n = self.process_info["N_n"]
        N_s = self.process_info["N_s"]
        N_p = self.process_info["N_p"]
        self.particles_n = np.empty(N_n, dtype=object)
        self.particles_p = np.empty(N_p, dtype=object)
        for indx in range(N_n):
            indx1 = indx2 = indx
            self.particles_n[indx] = ModParticle("particle_n_{}".format(indx), indx1, indx2, self.c,
                                                 self.y_avg, self.phi2, self.phi1_n, Ds_n, U_n, Parent=self)
        for indx in range(N_p):
            indx1 = indx
            indx2 = N_n + N_s + indx
            self.particles_p[indx] = ModParticle("particle_p_{}".format(indx), indx1, indx2, self.c,
                                                 self.y_avg, self.phi2, self.phi1_p, Ds_p, U_p, Parent=self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        pinfo = self.process_info
        # Thermal voltage = RT/F = kT/e = approximately 0.026 mV at room temp, 25 C
        V_thm = self.R() * self.T() / self.F()

        # We choose to use (cell centered) finite volume discretization for the electrolyte rather
        # than the built-in finite difference method for a few reasons
        # - It is a mass conservative method, which is important for quasi-neutral electrolyte models
        # - It is more stable at high electrolyte depletion than finite difference methods
        # As a result, we need some information about the domain on which we have discretized our
        # field variables
        # With the finite volume method, we store some information at cell centers
        # (scalar field variables like concentration and electric potential)
        # and store/calculate some information at the faces between cells
        # (fluxes like current density and flux of anions)
        # For more information, see, e.g.,
        #   http://www.ctcms.nist.gov/fipy/documentation/numerical/discret.html
        #   https://en.wikipedia.org/wiki/Finite_volume_method
        # Number of grid cells centers in each of the negative and positive electrodes
        N_n, N_p = self.x_centers_n.NumberOfPoints, self.x_centers_p.NumberOfPoints
        # Number of grid cell centers and faces along the entire electrode
        N_centers, N_faces = self.x_centers_full.NumberOfPoints, self.x_faces_full.NumberOfPoints
        # Number of grid cell centers along the separator
        N_s = N_centers - N_n - N_p
        # Coordinates of the cell centers and cell faces
        center_coords = np.array([self.xval_cells(indx) for indx in range(N_centers)])
        face_coords = np.array([self.xval_faces(indx) for indx in range(N_faces)])
        # Spacing between cell centers, which we will use for finite difference approximations for fluxes
        # We add space for a "ghost point" on each end, which will be used for boundary conditions.
        h_centers = np.hstack((np.diff(center_coords)[0], np.diff(center_coords), np.diff(center_coords)[-1]))
        # Spacing between cell faces.
        h_faces = np.diff(face_coords)

        # For convenience, make numpy arrays of variables at cell centers
        phi2 = np.array([self.phi2(indx) for indx in range(N_centers)])
        c = np.array([self.c(indx) for indx in range(N_centers)])
        dcdt = np.array([self.c.dt(indx) for indx in range(N_centers)])
        a = np.array([self.a_n()]*N_n + [Constant(0 * m**(-1))]*N_s + [self.a_p()]*N_p)
        j_p = np.array([self.particles_n[indx].j_p() for indx in range(N_n)]
                       + [Constant(0 * mol/(m**2 * s))]*N_s
                       + [self.particles_p[indx].j_p() for indx in range(N_p)])
        poros = np.array([self.poros_n()]*N_n + [self.poros_s()]*N_s + [self.poros_p()]*N_p)
        eff_factor_tmp = np.array([self.poros_n() / (self.poros_n()**self.BruggExp_n())]*(N_n+1)
                                  + [self.poros_s() / (self.poros_s()**self.BruggExp_s())]*N_s
                                  + [self.poros_p() / (self.poros_p()**self.BruggExp_p())]*(N_p+1))
        # The eff_factor is a prefactor for the transport in the porous medium compared to transport
        # in a free solution. It is needed at the cell faces because it is used in calculation of fluxes,
        # so we use a harmonic mean to approximate the value at the faces.
        eff_factor = (2*eff_factor_tmp[1:]*eff_factor_tmp[:-1]) / (eff_factor_tmp[1:] + eff_factor_tmp[:-1])

        # Boundary conditions on c and phi2 at current collectors.
        # For concentration, Thomas et al., Eq 15
        # For phi at current collectors, grad(phi) = 0 is required such that i2 = 0 at the current collectors
        # To do these, create "ghost points" on the end of cell-center vectors
        ctmp = np.empty(N_centers + 2, dtype=object)
        ctmp[1:-1] = c
        phi2tmp = np.empty(N_centers + 2, dtype=object)
        phi2tmp[1:-1] = phi2
        # No ionic current passes into the current collectors, which requires
        # grad(c) = grad(phi2) = 0
        # at both current collectors. We apply this by using the ghost points.
        ctmp[0] = ctmp[1]
        ctmp[-1] = ctmp[-2]
        phi2tmp[0] = phi2tmp[1]
        phi2tmp[-1] = phi2tmp[-2]
        # We'll need the value of c at the faces as well. We use a harmonic mean.
        c_faces = (2*ctmp[1:]*ctmp[:-1])/(ctmp[1:] + ctmp[:-1])

        # Approximate the gradients of these field variables at the faces
        dc = np.diff(ctmp) / h_centers
        dlogc = np.diff(np.log(ctmp / self.c_ref())) / h_centers
        dphi2 = np.diff(phi2tmp) / h_centers

        # Effective transport properties are required at faces between cells
        # Thomas et al., below Eq 3
        D_eff = eff_factor * D(c_faces / self.c_ref())
        kappa_eff = eff_factor * kappa(c_faces / self.c_ref())

        # Flux of charge (current density) at faces
        # Thomas et al., Eq 3
        i = -kappa_eff * (dphi2 - 2*V_thm*(1 - t_p(c_faces))*thermodynamic_factor(c_faces)*dlogc)

        # Flux of anions at faces
        # Based on Thomas et al., Eq 8
        # Using Thomas et al. Eq 9 and 10, using z_m = -1 and the paragraph below Eq 12 with Eq 13
        N_m = -D_eff*dc - (1 - t_p(c_faces)) * i / self.F()

        # Store values for the current density
        for indx in range(N_faces):
            eq = self.CreateEquation("i2_{}".format(indx))
            eq.Residual = self.i2(indx) - i[indx]

        # Divergence of fluxes
        di = np.diff(i) / h_faces
        dN_m = np.diff(N_m) / h_faces
        # Electrolyte: mass and charge conservation
        for indx in range(N_centers):
            # Thomas et al., Eq 11
            # Used instead of Eq 12, which is equivalent, noting that c_m = c for the electrolyte
            eq = self.CreateEquation("mass_cons_m_{}".format(indx), "anion mass conservation")
            eq.Residual = poros[indx]*dcdt[indx] + dN_m[indx]
            # Thomas et al., Eq 27 and 28
            eq = self.CreateEquation("charge_cons_{}".format(indx), "charge conservation")
            eq.Residual = -di[indx] - self.F()*a[indx]*j_p[indx]

        # Arbitrary datum for electric potential.
        # Thomas et al., below Eq 3
        # We apply this in the electrolyte at an arbitrary location, the negative current collector
        eq = self.CreateEquation("phi2_datum")
        eq.Residual = self.phiCC_n()

        # Electrode: charge conservation
        phi1_n = np.array([self.phi1_n(indx) for indx in range(N_n)])
        phi1_p = np.array([self.phi1_p(indx) for indx in range(N_p)])
        # We assume infinite conductivity in the electron conducting phase for simplicity
        # negative
        for indx in range(N_n):
            eq = self.CreateEquation("phi1_n_{}".format(indx))
            eq.Residual = phi1_n[indx] - self.phiCC_n()
        for indx in range(N_p):
            eq = self.CreateEquation("phi1_p_{}".format(indx))
            eq.Residual = phi1_p[indx] - self.phiCC_p()

        # Set the solid average filling fraction to non-changing in the separator region.
        # The variable shouldn't actually be defined in the separator, but it's convenient
        # for plotting purposes that it be defined over the full domain, so we simply fix its value
        # in the separator.
        for indx in range(N_n, N_n + N_s):
            eq = self.CreateEquation("y_avg_s_{}".format(indx))
            eq.Residual = self.y_avg.dt(indx)

        # Define the total current.
        # There are multiple ways to do this. Here, we set the current to be the (negative of the)
        # integral of the reaction rate into all the particles in the negative electrode.
        # This is equivalent to setting it equal to
        # - the integral the reaction rate into all the particles in the positive electrode
        # - the current density in the electrolyte in the separator (which must be uniform)
        # - the current density in the solid bulk electrode at the current collector (if using
        #   finite conductivity in the electron-conducting phase)
        eq = self.CreateEquation("Total_Current")
        eq.Residual = self.current() + np.sum(self.F()*a[:N_n]*j_p[:N_n]*h_centers[:N_n])

        # Define the measured voltage
        # Thomas et al., below Eq 4
        eq = self.CreateEquation("Voltage")
        eq.Residual = self.V() - (self.phiCC_p() - self.phiCC_n())

        # For the simulation, we can either specify the voltage and let current be a calculated output,
        # or we can specify the current and let voltage be a calculated output.
        # These correspond to CV (constant voltage) and CC (constant current) operation respectively.
        # We ramp quickly from an equilibrium to the set point to facilitate the numerical calculation
        # of consistent initial conditions.
        if pinfo["profileType"] == "CC":
            # Total Current Constraint Equation
            eq = self.CreateEquation("Total_Current_Constraint")
            eq.Residual = self.current() - self.currset()*(1 - np.exp(-Time()/self.tau_ramp()))
        elif pinfo["profileType"] == "CV":
            # Keep applied potential constant
            eq = self.CreateEquation("applied_potential")
            eq.Residual = self.V() - self.Vset()*(1 - np.exp(-Time()/self.tau_ramp()))

class SimBattery(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.F = 96485.34 * A*s/mol
        # Define the model we're going to simulate
        # Constant current (CC) or constant voltage (CV) simulation
        profileType = "CC"
        # Time at which to stop the simulation (used for CV simulations)
        tend = 2*3200e0 * s
        # Fraction of battery capacity to (dis)charge (used for CC simulations)
        capfrac = 0.90
        # Applied current or voltage (used in CC and CV simulations respectively)
        self.currset = 3e+1 * A/m**2
        self.Vset = 3.5 * V
        # Lenghts of regions of battery, negative electrode, separator, positive electrode
        self.L_n = 243e-6 * m
        self.L_s = 50e-6 * m
        self.L_p = 200e-6 * m
        # Porosies in each region
        self.poros_n = 0.3
        self.poros_s = 1.0
        self.poros_p = 0.3
        self.L_tot = self.L_n + self.L_s + self.L_p
        # Number of grid points in each region
        self.N_n = 20
        self.N_s = 20
        self.N_p = 20
        # Number of radial grid points for active particles in each electrode
        self.NR_n = 15
        self.NR_p = 15
        # Radius of active particles in each electrode
        self.R_n = 18e-6 * m
        self.R_p = 1e-6 * m
        # Maximum concentration of Li in the active materials
        self.csmax_n = 13.2e3 * mol/m**3
        self.csmax_p = 23.72e3 * mol/m**3
        # Initial filling fraction of each electrode.
        # For discharge, negative starts full and positive empty, opposite for charge
        self.ff0_n = 0.99
        self.ff0_p = 0.21
        # Capacity of each electrode
        capacity_n = self.csmax_n*(1 - self.poros_n)*self.L_n*self.F
        capacity_p = self.csmax_p*(1 - self.poros_p)*self.L_p*self.F
        # Capacity of battery
        capacity = min(capacity_n, capacity_p)
        if profileType == "CC":
            tend = capfrac * capacity / self.currset
        self.process_info = {"profileType": profileType, "tend": tend}

        self.process_info["N_n"] = self.N_n
        self.process_info["N_s"] = self.N_s
        self.process_info["N_p"] = self.N_p
        self.m = ModCell("tutorial_che_6", process_info=self.process_info)

    def SetUpParametersAndDomains(self):
        h_n = self.L_n / self.N_n
        h_s = self.L_s / self.N_s
        h_p = self.L_p / self.N_p
        xvec_centers_n = [h_n*(0.5 + indx) for indx in range(self.N_n)]
        xvec_centers_s = [self.L_n + h_s*(0.5 + indx) for indx in range(self.N_s)]
        xvec_centers_p = [(self.L_n + self.L_s) + h_p*(0.5 + indx) for indx in range(self.N_p)]
        xvec_centers = xvec_centers_n + xvec_centers_s + xvec_centers_p
        xvec_faces = [0 * m] + [h_n*(1 + indx) for indx in range(self.N_n)]
        xvec_faces += [self.L_n + h_s*(1 + indx) for indx in range(self.N_s)]
        xvec_faces += [(self.L_n + self.L_s) + h_p*(1 + indx) for indx in range(self.N_p)]
        # Domains in ModCell
        self.m.x_centers_n.CreateStructuredGrid(self.N_n - 1, 0, 1)
        self.m.x_centers_p.CreateStructuredGrid(self.N_p - 1, 0, 1)
        self.m.x_centers_full.CreateStructuredGrid(self.N_n + self.N_s + self.N_p - 1, 0, 1)
        self.m.x_faces_full.CreateStructuredGrid(self.N_n + self.N_s + self.N_p, 0, 1)
        self.m.x_centers_n.Points = [x.value for x in xvec_centers_n]
        self.m.x_centers_p.Points = [x.value for x in xvec_centers_p]
        self.m.x_centers_full.Points = [x.value for x in xvec_centers]
        self.m.x_faces_full.Points = [x.value for x in xvec_faces]
        # Domains in each particle
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            self.m.particles_n[indx_n].r.CreateStructuredGrid(self.NR_n - 1, 0, self.R_n.value)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
            self.m.particles_p[indx_p].r.CreateStructuredGrid(self.NR_p - 1, 0, self.R_p.value)
        # Parameters in ModCell
        self.m.F.SetValue(self.F)
        self.m.R.SetValue(8.31447 * J/(mol*K))
        self.m.T.SetValue(298 * K)
        self.m.L_n.SetValue(self.L_n)
        self.m.L_s.SetValue(self.L_s)
        self.m.L_p.SetValue(self.L_p)
        self.m.BruggExp_n.SetValue(-0.5)
        self.m.BruggExp_s.SetValue(-0.5)
        self.m.BruggExp_p.SetValue(-0.5)
        self.m.poros_n.SetValue(self.poros_n)
        self.m.poros_s.SetValue(self.poros_s)
        self.m.poros_p.SetValue(self.poros_p)
        self.m.a_n.SetValue((1-self.m.poros_n.GetQuantity())*3/self.R_n)
        self.m.a_p.SetValue((1-self.m.poros_p.GetQuantity())*3/self.R_p)
        self.m.c_ref.SetValue(1000 * mol/m**3)
        self.m.currset.SetValue(self.currset)
        self.m.Vset.SetValue(self.Vset)
        self.m.tau_ramp.SetValue(1e-3 * self.process_info["tend"])
        self.m.xval_cells.SetValues(np.array(xvec_centers))
        self.m.xval_faces.SetValues(np.array(xvec_faces))
        # Parameters in each particle
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_n)
            p.V_thermal.SetValue(self.m.R.GetQuantity()*self.m.T.GetQuantity()/self.m.F.GetQuantity())
            p.R.SetValue(self.R_n)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
            p = self.m.particles_p[indx_p]
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_p)
            p.V_thermal.SetValue(self.m.R.GetQuantity()*self.m.T.GetQuantity()/self.m.F.GetQuantity())
            p.R.SetValue(self.R_p)

    def SetUpVariables(self):
        cs0_n = self.ff0_n * self.csmax_n
        cs0_p = self.ff0_p * self.csmax_p
        # ModCell
        for indx in range(self.m.x_centers_full.NumberOfPoints):
            self.m.c.SetInitialCondition(indx, 1e3 * mol/m**3)
            if indx < self.N_n:
                self.m.y_avg.SetInitialCondition(indx, self.ff0_n)
            elif indx < self.N_n + self.N_s:
                self.m.y_avg.SetInitialCondition(indx, 0.)
            elif indx < self.N_n + self.N_s + self.N_p:
                self.m.y_avg.SetInitialCondition(indx, self.ff0_p)
        self.m.phi1_n.SetInitialGuesses(U_n(self.ff0_n))
        self.m.phi1_p.SetInitialGuesses(U_p(self.ff0_p))
        self.m.phiCC_n.SetInitialGuess(U_n(self.ff0_n))
        self.m.phiCC_p.SetInitialGuess(U_p(self.ff0_p))
        # particles
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, cs0_n)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
            p = self.m.particles_p[indx_p]
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, cs0_p)

# Use daeSimulator class
def guiRun(app):
    sim = SimBattery()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = SimBattery()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = simulation.process_info["tend"].value / 100
    simulation.TimeHorizon = simulation.process_info["tend"].value

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) is False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()

    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
