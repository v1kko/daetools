#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           flux_limiters.py
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

MIN     = Min
MAX     = Max
FABS    = Abs
beta    = 1.5 # For Sweby and Osher (1 <= beta  <= 2)
theta   = 1.5 # For van Leer minmod (1 <= theta <= 2)

class HRFluxLimiter(object):
    def __init__(self, phi_callable, r_epsilon = 1e-10):
        if not phi_callable:
            raise RuntimeError('Invalid flux limiter function specified')

        self.Phi     = phi_callable
        self.epsilon = r_epsilon

    def r(self, i, ni):
        if hasattr(ni, '__call__'): # Should be daetools daeVariable
            return (ni(i+1) - ni(i) + self.epsilon) / (ni(i) - ni(i-1) + self.epsilon)

        else: # Python list or nd_array
            return (ni[i+1] - ni[i] + self.epsilon) / (ni[i] - ni[i-1] + self.epsilon)

    def ni_edge_plus(self, i, ni, nL):
        if i == 0:      # Right face of the first cell: central interpolation (k=1)
            return 0.5 * (ni(0) + ni(1))
        elif i == nL-1: # Right face of the last cell: one-sided upwind scheme (k=-1)
            return ni(i) + 0.5 * (ni(i) - ni(i-1))
        else:           # Other cells: k=1/3
            return ni(i) + 0.5 * self.Phi(self.r(i, ni)) * (ni(i) - ni(i-1))

def Phi_CHARM(r):
    """CHARM"""
    return r * (3*r+1) / ((r+1)**2)

def Phi_HCUS(r):
    """HCUS"""
    return 1.5 * (r + FABS(r)) / (r + 2)

def Phi_HQUICK(r):
    """HQUICK"""
    return 2.0 * (r + FABS(r)) / (r + 3)

def Phi_Koren(r):
    """Koren"""
    #r = 1.0/r
    return MAX(0.0, MIN(2.0*r, MIN(1.0/3.0 + 2.0*r/3, 2.0)))

def Phi_minmod(r):
    """minmod"""
    return MAX(0.0, MIN(1.0, r))

def Phi_monotinized_central(r):
    """MC"""
    return MAX(0.0, MIN(2.0*r, MIN(0.5*(1+r), 2.0)))

def Phi_Osher(r):
    """Osher"""
    # 1 <= beta <= 2
    return MAX(0.0, MIN(r, beta))

def Phi_ospre(r):
    """ospre"""
    return 1.5 * (r*r + r) / (r*r + r + 1.0)

def Phi_smart(r):
    """smart"""
    return MAX(0.0, MIN(2.0*r, MIN(0.25+0.75*r, 4.0)))

def Phi_superbee(r):
    """superbee"""
    return MAX(0.0, MAX(MIN(2.0*r, 1.0), MIN(r, 2.0)))

def Phi_Sweby(r):
    """Sweby"""
    # 1 <= beta <= 2
    return MAX(0.0, MAX(MIN(beta*r, 1.0), MIN(r, beta)))

def Phi_UMIST(r):
    """UMIST"""
    return MAX(0.0, MIN(2.0*r, MIN(0.25+0.75*r, MIN(0.75+0.25*r, 2.0))))

def Phi_vanAlbada1(r):
    """vanAlbada1"""
    return (r*r + r) / (r*r + 1.0)

def Phi_vanAlbada2(r):
    """vanAlbada2"""
    return (2.0*r) / (r*r + 1.0)

def Phi_vanLeer(r):
    """vanLeer"""
    return (r + FABS(r)) / (1.0 + FABS(r))

def Phi_vanLeer_minmod(r):
    """vanLeerMinmod"""
    # 1 <= theta <= 2
    return MAX(0.0, MIN(theta*r, MIN(0.5*(1.0+r), theta)))

supported_schemes = [Phi_HCUS, Phi_HQUICK, Phi_Koren, Phi_monotinized_central,
                     Phi_minmod, Phi_Osher, Phi_ospre, Phi_smart, Phi_superbee,
                     Phi_Sweby, Phi_UMIST, Phi_vanAlbada1, Phi_vanAlbada2, Phi_vanLeer,
                     Phi_vanLeer_minmod]

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
    for scheme in supported_schemes:
        if scheme == Phi_Koren:
            phis[:] = [scheme(r) for r in rs_koren]
        else:
            phis[:] = [scheme(r) for r in rs]

        axes = matplotlib.pyplot.subplot(4, 4, counter)
        fp9  = matplotlib.font_manager.FontProperties(family='Cantarell', style='normal', variant='normal', weight='normal', size=9)
        fp11 = matplotlib.font_manager.FontProperties(family='Cantarell', style='normal', variant='normal', weight='normal', size=12)
        axes.plot(rs, phis, 'r-')
        axes.fill_between(tvdx, tvdy1, tvdy2, alpha = 0.3)
        axes.set_title(scheme.__doc__, fontproperties=fp11)
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
    MIN     = min
    MAX     = max
    FABS    = math.fabs
    beta    = 1.5 # For Sweby and Osher (1 <= beta  <= 2)
    theta   = 1.5 # For van Leer minmod (1 <= theta <= 2)

    plot_flux_limiters()
    sys.exit()
