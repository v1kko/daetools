#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""***********************************************************************************
                           hr_upwind_scheme.py
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
************************************************************************************"""
import sys, numpy, math
from daetools.pyDAE import *

# These function operate on adouble objects
MIN     = Min
MAX     = Max
FABS    = Abs
beta    = 1.5 # For Sweby and Osher (1 <= beta  <= 2)
theta   = 1.5 # For van Leer minmod (1 <= theta <= 2)

class daeHRUpwindSchemeEquation(object):
    supported_flux_limiters = []
    
    def __init__(self, variable, domain, phi, r_epsilon = 1e-10, reversedFlow = False):
        """
        """
        if not callable(phi):
            raise RuntimeError('Invalid flux limiter function specified (must be a callable)')
        if not callable(variable):
            raise RuntimeError('Invalid variable specified (must be a daeVariable object or a callable)')
        if not callable(domain):
            raise RuntimeError('Invalid domain specified (must be a daeDomain object or a callable)')

        self.c            = variable  # daeVariable object (or a callable)
        self.x            = domain    # daeDomain object
        self.phi          = phi       # Flux limiter function (a callable)
        self.r_epsilon    = r_epsilon # epsilon in the upwind ratio (r) expression
        self.reversedFlow = reversedFlow 

    def dc_dt(self, i, variable = None):
        """Accumulation term in the cell-centered finite-volume discretisation:
            
           :math:`\int_{\Omega_i} {\partial c_i \over \partial t} dx`
        """
        x     = self.x
        Nx    = self.x.NumberOfPoints
        if variable:
            if not callable(variable):
                raise RuntimeError('Invalid variable specified (must be a daeVariable object or a callable)')
            c = variable
        else:
            c = self.c
        dc_dt = lambda j: dt(c(j))
        
        if not self.reversedFlow:
            if i == 0: # Sto ovde imam za 0??
                return (x[i+1]-x[i]) * dc_dt(i)
            else:
                return (x[i]-x[i-1]) * dc_dt(i)
        else: # reversible flow
            if i == Nx-1: # Sto ovde imam za Nx-1??
                return (x[Nx-1]-x[Nx-2]) * dc_dt(Nx-1)
            else:
                return (x[i]-x[i+1]) * dc_dt(i)

    def dc_dx(self, i, S = None, variable = None):
        """Convection term in the cell-centered finite-volume discretisation:
            
           :math:`c_{i + {1 \over 2}} - c_{i - {1 \over 2}}`.
           
           Cell-face state :math:`c_{i+{1 \over 2}}` is given as:
               
           :math:`{c}_{i + {1 \over 2}} = c_i  + \phi \left( r_{i + {1 \over 2}} \\right) \left( c_i - c_{i-1}  \\right)`
           
           where :math:`\phi` is the flux limiter function and :math:`r_{i + {1 \over 2}}` the upwind ratio of consecutive 
           solution gradients:
               
           :math:`r_{i + {1 \over 2}} = {{c_{i+1} - c_{i} + \epsilon} \over {c_{i} - c_{i-1} + \epsilon}}`.
           
           If the source term integral :math:`S= {1 \over u} \int_{\Omega_i} s(x) dx` is not ``None`` then the convection term is given as:
           :math:`(c-S)_{i + {1 \over 2}} - (c-S)_{i - {1 \over 2}}`.
        """
        if variable:
            if not callable(variable):
                raise RuntimeError('Invalid variable specified (must be a daeVariable object or a callable)')
            c = variable
        else:
            c = self.c
        
        if not self.reversedFlow:
            return self._c_edge_plus(i, c, S)     - self._c_edge_plus(i-1, c, S)
        else: # reversible flow
            return self._c_edge_plus_rev(i, c, S) - self._c_edge_plus_rev(i+1, c, S)
            
    def d2c_dx2(self, i, variable = None):
        """Diffusion term in the cell-centered finite-volume discretisation:           
           
           :math:`\left( \partial c_i \over \partial x \\right)_{i + {1 \over 2}} - \left( \partial c_i \over \partial x \\right)_{i - {1 \over 2}}`
        """
        if variable:
            if not callable(variable):
                raise RuntimeError('Invalid variable specified (must be a daeVariable object or a callable)')
            c = variable
        else:
            c = self.c

        if not self.reversedFlow:
            return self._dc_dx_edge_plus(i, c)     - self._dc_dx_edge_plus(i-1, c)
        else: # reversible flow
            return self._dc_dx_edge_plus_rev(i, c) - self._dc_dx_edge_plus_rev(i+1, c)
    
    def source(self, s, i):
        """Source term in the cell-centered finite-volume discretisation: 
            
           :math:`\int_{\Omega_i} s(x) dx`
        """
        # The cell average shouldn't depend on a flow direction.
        # Anyhow, everything is fine as long as the grid is uniform.
        return self._AverageOverCell(s,i) 

    ########################################################################
    # Implementation details
    ########################################################################
    def _r(self, i, S = None):
        # Upwind ratio of consecutive solution gradients.
        # It may include the source term integral S(x).
        eps = self.r_epsilon
        c   = self.c
        cs  = lambda j: c(j)-S(j) if S else c(j)
            
        return (cs(i+1) - cs(i) + eps) / (cs(i) - cs(i-1) + eps)
    
    def _r_rev(self, i, S = None):
        # Upwind ratio of consecutive solution gradients for the reversed flow.
        # It may include the source term integral S(x).
        eps = self.r_epsilon
        c   = self.c
        cs = lambda j: c(j)-S(j) if S else c(j)
            
        return (cs(i-1) - cs(i) + eps) / (cs(i) - cs(i+1) + eps)

    def _c_edge_plus(self, i, c, S):
        # c at the i+1/2 face (cell outlet)
        phi = self.phi
        r   = self._r
        x   = self.x
        Nx  = self.x.NumberOfPoints
        cs  = lambda j: c(j)-S(j) if S else c(j)

        if i == 0:      
            # Right face of the first cell: central interpolation (k=1)
            return 0.5 * (cs(0) + cs(1))
        elif i == Nx-1: 
            # Right face of the last cell: one-sided upwind scheme (k=-1)
            return cs(i) + 0.5 * (cs(i) - cs(i-1))
        elif i > 0 and i < Nx-1:           
            # Other cells: k=1/3
            return cs(i) + 0.5 * phi(r(i,S)) * (cs(i) - cs(i-1))
        else:
            raise RuntimeError('c_edge_plus: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    def _c_edge_plus_rev(self, i, c, S):
        # c at the i+1/2 face (cell outlet) for the reversed flow
        phi = self.phi
        r   = self._r_rev
        x   = self.x
        Nx  = self.x.NumberOfPoints
        cs  = lambda j: c(j)-S(j) if S else c(j)
        
        if i == 0:
            # Left face of the last cell (first in the reversed):
            return cs(i) + 0.5 * (cs(i) - cs(i+1))
        elif i == Nx-1: 
            # Left face of the first cell (last in the reversed):
            return 0.5 * (cs(Nx-1) + cs(Nx-2))
        elif i > 0 and i < Nx-1:           
            # Other cells: k=1/3
            return cs(i) + 0.5 * phi(r(i,S)) * (cs(i) - cs(i+1))
        else:
            raise RuntimeError('c_edge_plus_rev: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    def _dc_dx_edge_plus(self, i, c):
        # Diffusion at the i+1/2 face (cell outlet)
        x   = self.x
        Nx  = self.x.NumberOfPoints

        if i == 0:
            # Right face of the first cell: biased central-difference
            return (-8*c(i) + 9*c(i+1) + c(i+2)) / (3 * (x[i+1] - x[i]))
        elif i == Nx-1:
            # Right face of the last cell: biased central-difference
            return (8*c(i) - 9*c(i-1) + c(i-2)) / (3 * (x[i] - x[i-1]))
        elif i > 0 and i < Nx-1:           
            # Other cells: central-difference O(h^2)
            return (c(i+1) - c(i)) / (x[i+1] - x[i])
        else:
            raise RuntimeError('dc_dx_edge_plus: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    def _dc_dx_edge_plus_rev(self, i, c):
        # Diffusion at the i+1/2 face (cell outlet) for the reversed flow
        x   = self.x
        Nx  = self.x.NumberOfPoints

        if i == 0:
            # Left face of the last cell: biased central-difference
            return (8*c(i) - 9*c(i+1) + c(i+2)) / (3 * (x[i+1] - x[i]))
        elif i == Nx-1:
            # Left face of the first cell: biased central-difference
            return (-8*c(i) + 9*c(i-1) + c(i-2)) / (3 * (x[i] - x[i-1]))
        elif i > 0 and i < Nx-1:           
            # Other cells: central-difference O(h^2)
            return (c(i) - c(i+1)) / (x[i+1] - x[i])
        else:
            raise RuntimeError('dc_dx_edge_plus_rev: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    def _AverageOverCell(self, f, i):
        """                                   i+1/2
        Integral over a cell i: S = integral      (f(x)*dx)
                                              i-1/2

        We could use a simple trapezoidal rule: (x      - x     ) * [f(x     ) + f(x     )] / 2
                                                  i+1/2    i-1/2        i+1/2       i-1/2
        or to assume the source everywhere in the cell is equal to the value at the plus edge:
                                                (x      - x     ) * f(x     )   
                                                  i+1/2    i-1/2       i+1/2
        The latter gives better results (double check why).
        """
        x   = self.x
        xp  = self.x.Points
        Nx  = self.x.NumberOfPoints
        if i == 0:
            # The first cell
            return f(i) * (xp[i+1] - xp[i])
        elif i == Nx-1:
            # The last cell
            return f(i) * (xp[i] - xp[i-1]) 
        elif i > 0 and i < Nx-1:           
            # Other cells
            return f(i) * (xp[i] - xp[i-1])
        else:
            raise RuntimeError('VolumeAverage: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    @staticmethod
    def Phi_CHARM(r):
        """CHARM"""
        return r * (3*r+1) / ((r+1)**2)

    @staticmethod
    def Phi_HCUS(r):
        """HCUS"""
        return 1.5 * (r + FABS(r)) / (r + 2)

    @staticmethod
    def Phi_HQUICK(r):
        """HQUICK"""
        return 2.0 * (r + FABS(r)) / (r + 3)

    @staticmethod
    def Phi_Koren(r):
        """Koren"""
        #r = 1.0/r
        return MAX(0.0, MIN(2.0*r, MIN(1.0/3.0 + 2.0*r/3, 2.0)))

    @staticmethod
    def Phi_minmod(r):
        """minmod"""
        return MAX(0.0, MIN(1.0, r))

    @staticmethod
    def Phi_monotinized_central(r):
        """MC"""
        return MAX(0.0, MIN(2.0*r, MIN(0.5*(1+r), 2.0)))

    @staticmethod
    def Phi_Osher(r):
        """Osher"""
        # 1 <= beta <= 2
        return MAX(0.0, MIN(r, beta))

    @staticmethod
    def Phi_ospre(r):
        """ospre"""
        return 1.5 * (r*r + r) / (r*r + r + 1.0)

    @staticmethod
    def Phi_smart(r):
        """smart"""
        return MAX(0.0, MIN(2.0*r, MIN(0.25+0.75*r, 4.0)))

    @staticmethod
    def Phi_superbee(r):
        """superbee"""
        return MAX(0.0, MAX(MIN(2.0*r, 1.0), MIN(r, 2.0)))

    @staticmethod
    def Phi_Sweby(r):
        """Sweby"""
        # 1 <= beta <= 2
        return MAX(0.0, MAX(MIN(beta*r, 1.0), MIN(r, beta)))

    @staticmethod
    def Phi_UMIST(r):
        """UMIST"""
        return MAX(0.0, MIN(2.0*r, MIN(0.25+0.75*r, MIN(0.75+0.25*r, 2.0))))

    @staticmethod
    def Phi_vanAlbada1(r):
        """vanAlbada1"""
        return (r*r + r) / (r*r + 1.0)

    @staticmethod
    def Phi_vanAlbada2(r):
        """vanAlbada2"""
        return (2.0*r) / (r*r + 1.0)

    @staticmethod
    def Phi_vanLeer(r):
        """vanLeer"""
        return (r + FABS(r)) / (1.0 + FABS(r))

    @staticmethod
    def Phi_vanLeer_minmod(r):
        """vanLeerMinmod"""
        # 1 <= theta <= 2
        return MAX(0.0, MIN(theta*r, MIN(0.5*(1.0+r), theta)))

daeHRUpwindSchemeEquation.supported_flux_limiters = [daeHRUpwindSchemeEquation.Phi_HCUS, 
                                                     daeHRUpwindSchemeEquation.Phi_HQUICK, 
                                                     daeHRUpwindSchemeEquation.Phi_Koren, 
                                                     daeHRUpwindSchemeEquation.Phi_monotinized_central,
                                                     daeHRUpwindSchemeEquation.Phi_minmod, 
                                                     daeHRUpwindSchemeEquation.Phi_Osher, 
                                                     daeHRUpwindSchemeEquation.Phi_ospre, 
                                                     daeHRUpwindSchemeEquation.Phi_smart, 
                                                     daeHRUpwindSchemeEquation.Phi_superbee,
                                                     daeHRUpwindSchemeEquation.Phi_Sweby, 
                                                     daeHRUpwindSchemeEquation.Phi_UMIST, 
                                                     daeHRUpwindSchemeEquation.Phi_vanAlbada1, 
                                                     daeHRUpwindSchemeEquation.Phi_vanAlbada2, 
                                                     daeHRUpwindSchemeEquation.Phi_vanLeer,
                                                     daeHRUpwindSchemeEquation.Phi_vanLeer_minmod]

def plot_flux_limiters():
    import matplotlib.pyplot
    n = 30
    figure = matplotlib.pyplot.figure(figsize = (9,9), dpi=100, facecolor='white')
    figure.canvas.set_window_title('Flux Limiters')

    tvdx  = numpy.zeros(n)
    tvdy1 = numpy.zeros(n)
    tvdy2 = numpy.zeros(n)
    tvdx[0:5]   = numpy.linspace(1e-10, 0.5,  5)
    tvdx[5:10]  = numpy.linspace(0.5,   1.0,  5)
    tvdx[10:20] = numpy.linspace(1.0,   2.0, 10)
    tvdx[20:30] = numpy.linspace(2.0,   3.0, 10)

    tvdy1[0:5]   = numpy.linspace(0.0, 1.0,  5)
    tvdy1[5:10]  = numpy.linspace(1.0, 1.0,  5)
    tvdy1[10:20] = numpy.linspace(1.0, 2.0, 10)
    tvdy1[20:30] = numpy.linspace(2.0, 2.0, 10)

    tvdy2[0:5]   = numpy.linspace(0.0, 0.5,  5)
    tvdy2[5:10]  = numpy.linspace(0.5, 1.0,  5)
    tvdy2[10:20] = numpy.linspace(1.0, 1.0, 10)
    tvdy2[20:30] = numpy.linspace(1.0, 1.0, 10)

    rs       = numpy.linspace(1e-10, 3.0, n)
    rs_koren = rs
    phis     = numpy.zeros(n)

    counter = 1
    for fl in daeHRUpwindScheme.supported_flux_limiters:
        if fl == daeHRUpwindScheme.Phi_Koren:
            phis[:] = [fl(r) for r in rs_koren]
        else:
            phis[:] = [fl(r) for r in rs]

        axes = matplotlib.pyplot.subplot(4, 4, counter)
        fp9  = matplotlib.font_manager.FontProperties(family='Cantarell', style='normal', variant='normal', weight='normal', size=9)
        fp11 = matplotlib.font_manager.FontProperties(family='Cantarell', style='normal', variant='normal', weight='normal', size=12)
        axes.plot(rs, phis, 'r-')
        axes.fill_between(tvdx, tvdy1, tvdy2, alpha = 0.3)
        axes.set_title(fl.__doc__, fontproperties=fp11)
        axes.set_xlim((0.0, 3.0))
        axes.set_ylim((0.0, 3.0))
        axes.set_xticks([0, 1, 2, 3])
        axes.set_yticks([0, 1, 2, 3])
        axes.set_ylabel('$\phi( r )$', fontproperties=fp11)
        axes.set_xlabel('$r$', fontproperties=fp11)
        axes.grid(True)
        counter += 1
        for xlabel in axes.get_xticklabels():
            xlabel.set_fontproperties(fp9)
        for ylabel in axes.get_yticklabels():
            ylabel.set_fontproperties(fp9)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

if __name__ == "__main__":
    # Change the functions definitions to work with floats
    MIN     = min
    MAX     = max
    FABS    = math.fabs
    beta    = 1.5 # For Sweby and Osher (1 <= beta  <= 2)
    theta   = 1.5 # For van Leer minmod (1 <= theta <= 2)

    plot_flux_limiters()
