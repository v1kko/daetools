#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
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
************************************************************************************
"""
import sys, numpy, math
import matplotlib.pyplot
from daetools.pyDAE import *

# These function operate on adouble objects
MIN     = Min
MAX     = Max
FABS    = Abs
beta    = 1.5 # For Sweby and Osher (1 <= beta  <= 2)
theta   = 1.5 # For van Leer minmod (1 <= theta <= 2)

class daeHRUpwindScheme(object):
    supported_flux_limiters = []
    
    def __init__(self, c, x, phi, r_epsilon = 1e-10):
        if not callable(phi):
            raise RuntimeError('Invalid flux limiter function specified (must be a callable)')
        if not callable(c):
            raise RuntimeError('Invalid variable specified (must be daeVariable object or a callable)')
        if not callable(x):
            raise RuntimeError('Invalid domain specified (must be daeDomain object)')

        self.c         = c         # daeVariable object (or a callable)
        self.x         = x         # daeDomain object
        self.phi       = phi       # Flux limiter function (a callable)
        self.r_epsilon = r_epsilon # epsilon in the upwind ratio (r) expression

    def dc_dt(self, i):
        # Accumulation term in the cell-centered finite-volume discretisation
        x     = self.x
        c     = self.c
        dc_dt = lambda i: dt(c(i))  
        return (x[i]-x[i-1]) * dc_dt(i)
    
    def dc_dx(self, i, S = None):
        # Convection term in the cell-centered finite-volume discretisation
        return self.c_edge_plus(i,S) - self.c_edge_plus(i-1,S)
    
    def d2c_dx2(self, i):
        # Diffusion term in the cell-centered finite-volume discretisation
        return self.dc_dx_edge_plus(i) - self.dc_dx_edge_plus(i-1)

    def S_integral(self, s, i, u):
        # Integral of the source term: S(x) = 1/u * Integral[s(x)*dx]
        return (1 / u(i)) * self._AverageOverCell(s, i)

    ########################################################################
    # Implementation details
    ########################################################################
    def r(self, i, S = None):
        # Upwind ratio of consecutive solution gradients
        # It may include the source term integral S(x)
        eps = self.r_epsilon
        c   = self.c
        cs = lambda i: c(i)-S(i) if S else c(i)
            
        return (cs(i+1) - cs(i) + eps) / (cs(i) - cs(i-1) + eps)
    
    def c_edge_plus(self, i, S = None):
        # c at the i+1/2 face (cell outlet)
        phi = self.phi
        r   = self.r
        x   = self.x
        Nx  = self.x.NumberOfPoints
        c   = self.c
        cs = lambda i: c(i)-S(i) if S else c(i)
        
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

    def dc_dx_edge_plus(self, i):
        # Diffusion at the i+1/2 face (cell outlet)
        x   = self.x
        Nx  = self.x.NumberOfPoints
        c   = self.c

        if i == 0:
            # Right face of the first cell: biased central-difference
            return (-8*c(0) + 9*c(1) + c(2)) / (3 * (x[1] - x[0]))
        elif i == Nx-1:
            # Right face of the last cell: biased central-difference
            return (8*c(i) - 9*c(i-1) + c(i-2)) / (3 * (x[i] - x[i-1]))
        elif i > 0 and i < Nx-1:           
            # Other cells: central-difference O(h^2)
            return (c(i+1) - c(i)) / (x[i+1] - x[i])
        else:
            raise RuntimeError('dc_dx_edge_plus: Invalid index specified: %d (no. points is %d)' % (i, Nx))

    def _AverageOverCell(self, f, i):
        """                                   i+1/2
        Integral over a cell i: S = integral      (f(x)*dx)
                                              i-1/2

        Here we use a simple trapezoidal rule: (x      - x     ) * [f(x     ) + f(x     )] / 2
                                                 i+1/2    i-1/2        i+1/2       i-1/2
        """
        x   = self.x
        xp  = self.x.Points
        Nx  = self.x.NumberOfPoints
        if i == 0:
            # The first cell
            return 0.5 * (f(0) + f(1)) * (xp[1] - xp[0])
        elif i == Nx-1:
            # The last cell
            return 0.5 * (f(i) + f(i-1)) * (xp[i] - xp[i-1])
        elif i > 0 and i < Nx-1:           
            # Other cells
            return 0.5 * (f(i) + f(i-1)) * (xp[i] - xp[i-1])
        else:
            raise RuntimeError('VolumeAverage: Invalid index specified: %d (no. points is %d)' % (i, Nx))
        
    """
    def r_rev(self, i, c):
        if hasattr(c, '__call__'): # Should be daetools daeVariable
            return (c(i-1) - c(i) + self.epsilon) / (c(i) - c(i+1) + self.epsilon)

    def c_edge_plus_rev(self, i, c):
        if i == 0:      # Right face of the first cell: central interpolation (k=1)
            return 0.5 * (c(0) + c(1))
        elif i == self.Nx-1: # Right face of the last cell: one-sided upwind scheme (k=-1)
            return c(i) + 0.5 * (c(i) - c(i-1))
        else:           # Other cells: k=1/3
            return c(i) + 0.5 * self.phi(self.r(i, c)) * (c(i) - c(i-1))
    """
    
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

daeHRUpwindScheme.supported_flux_limiters = [daeHRUpwindScheme.Phi_HCUS, 
                                             daeHRUpwindScheme.Phi_HQUICK, 
                                             daeHRUpwindScheme.Phi_Koren, 
                                             daeHRUpwindScheme.Phi_monotinized_central,
                                             daeHRUpwindScheme.Phi_minmod, 
                                             daeHRUpwindScheme.Phi_Osher, 
                                             daeHRUpwindScheme.Phi_ospre, 
                                             daeHRUpwindScheme.Phi_smart, 
                                             daeHRUpwindScheme.Phi_superbee,
                                             daeHRUpwindScheme.Phi_Sweby, 
                                             daeHRUpwindScheme.Phi_UMIST, 
                                             daeHRUpwindScheme.Phi_vanAlbada1, 
                                             daeHRUpwindScheme.Phi_vanAlbada2, 
                                             daeHRUpwindScheme.Phi_vanLeer,
                                             daeHRUpwindScheme.Phi_vanLeer_minmod]

def plot_flux_limiters():
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
