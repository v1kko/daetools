/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DIURNAL_KINETICS_MODEL_H
#define DIURNAL_KINETICS_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <math.h>
#include <OpenCS/models/cs_number.h>
using namespace cs;

// An auxiliary class to handle the centered finite difference scheme on a 2D domain
class DiurnalKinetics_2D
{
public:
    DiurnalKinetics_2D(int nx, int ny, const csNumber_t& bc_c_flux):
        Nx(nx), Ny(ny), C_flux_bc(bc_c_flux)
    {
        x0 =  0.0;
        x1 = 20.0;
        y0 = 30.0;
        y1 = 50.0;
        dx = (x1-x0) / (Nx-1);
        dy = (y1-y0) / (Ny-1);

        x_domain.resize(Nx);
        y_domain.resize(Ny);
        for(int x = 0; x < Nx; x++)
            x_domain[x] = x0 + x * dx;
        for(int y = 0; y < Ny; y++)
            y_domain[y] = y0 + y * dy;

        C1_start_index = 0*Nx*Ny;
        C2_start_index = 1*Nx*Ny;

        C1_values = NULL;
        C2_values = NULL;

        Nequations = 2*Nx*Ny;

        V   =  1.00E-03;
        Kh  =  4.00E-06;
        Kv0 =  1.00E-08;
        q1  =  1.63E-16;
        q2  =  4.66E-16;
        C3  =  3.70E+16;
        a3  = 22.62;
        a4  =  7.601;
    }

    void SetInitialConditions(std::vector<real_t>& C0)
    {
        int ix, iy;
        C0.assign(2*Nx*Ny, 0.0);

        real_t* C1_0 = &C0[C1_start_index];
        real_t* C2_0 = &C0[C2_start_index];

        for(ix = 0; ix < Nx; ix++)
        {
            for(iy = 0; iy < Ny; iy++)
            {
                int index = getIndex(ix,iy);

                real_t x = x_domain[ix];
                real_t y = y_domain[iy];

                C1_0[index] = 1E6  * alfa(x) * beta(y);
                C2_0[index] = 1E12 * alfa(x) * beta(y);
            }
        }
    }

    void CreateEquations(const std::vector<csNumber_t>& C_values,
                         const csNumber_t& time,
                         std::vector<csNumber_t>& equations)
    {
        if(C_values.size() != Nequations)
            std::runtime_error("Invalid size of data arrays");

        C1_values = &C_values[C1_start_index];
        C2_values = &C_values[C2_start_index];

        equations.resize(Nequations);

        int eq = 0;
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                /* Component 1 */
                csNumber_t dC1_dt = V * dC1_dx(x,y) +                             /* x-axis convection term */
                                    Kh    * d2C1_dx2(x,y) +                       /* x-axis diffusion term  */
                                    Kv(y) * (0.2 * dC1_dy(x,y) + d2C1_dy2(x,y)) + /* y-axis diffusion term  */
                                    R1(C1(x,y), C2(x,y), time);                   /* generation term        */


                equations[eq++] = dC1_dt;
            }
        }
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                /* Component 2 */
                csNumber_t dC2_dt = V * dC2_dx(x,y) +                             /* x-axis convection term */
                                    Kh    * d2C2_dx2(x,y) +                       /* x-axis diffusion term  */
                                    Kv(y) * (0.2 * dC2_dy(x,y) + d2C2_dy2(x,y)) + /* y-axis diffusion term  */
                                    R2(C1(x,y), C2(x,y), time);                   /* generation term        */
                equations[eq++] = dC2_dt;
            }
        }
    }

protected:
    int getIndex(int x, int y)
    {
        if(x < 0 || x >= Nx)
            throw std::runtime_error("Invalid x index");
        if(y < 0 || y >= Ny)
            std::runtime_error("Invalid y index");
        return Ny*x + y;
    }

    real_t alfa(real_t x)
    {
        real_t xmid = (x1+x0)/2;
        real_t cx = 0.1 * (x - xmid);
        real_t cx2 = cx*cx;
        return 1 - cx2 + 0.5*cx2*cx2;
    }
    real_t beta(real_t y)
    {
        real_t ymid = (y1+y0)/2;
        real_t cy = 0.1 * (y - ymid);
        real_t cy2 = cy*cy;
        return 1 - cy2 + 0.5*cy2*cy2;
    }

    csNumber_t Kv(int y)
    {
        return Kv0 * std::exp(y_domain[y] / 5.0);
    }

    csNumber_t q3(const csNumber_t& time)
    {
        real_t w = 3.1415926535898 / 43200;
        csNumber_t sinwt = cs::max(0.0, cs::sin(w*time)) + 1E-20;
        return cs::exp(-a3 / sinwt);
    }

    csNumber_t q4(const csNumber_t& time)
    {
        real_t w = 3.1415926535898 / 43200;
        csNumber_t sinwt = cs::max(0.0, cs::sin(w*time)) + 1E-20;
        return cs::exp(-a4 / sinwt);
    }

    csNumber_t R1(const csNumber_t& c1, const csNumber_t& c2, const csNumber_t& time)
    {
        return -q1*c1*C3 - q2*c1*c2 + 2*q3(time)*C3 + q4(time)*c2;
    }

    csNumber_t R2(const csNumber_t& c1, const csNumber_t& c2, const csNumber_t& time)
    {
        return q1*c1*C3 - q2*c1*c2 - q4(time)*c2;
    }

    const csNumber_t& C1(int x, int y)
    {
        int index = getIndex(x, y);
        return C1_values[index];
    }
    const csNumber_t& C2(int x, int y)
    {
        int index = getIndex(x, y);
        return C2_values[index];
    }

    // First order partial derivative per x.
    csNumber_t dC1_dx(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 use the boundary value (C_flux_bc = 0.0 in this example).
        if(x == 0 || x == Nx-1)
            return C_flux_bc;

        const csNumber_t& ci1 = C1(x+1, y);
        const csNumber_t& ci2 = C1(x-1, y);
        return (ci1 - ci2) / (2*dx);
    }
    csNumber_t dC2_dx(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 use the boundary value (C_flux_bc = 0.0 in this example).
        if(x == 0 || x == Nx-1)
            return C_flux_bc;

        const csNumber_t& ci1 = C2(x+1, y);
        const csNumber_t& ci2 = C2(x-1, y);
        return (ci1 - ci2) / (2*dx);
    }

    // First order partial derivative per y.
    csNumber_t dC1_dy(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 use the boundary value (C_flux_bc = 0.0 in this example).
        if(y == 0 || y == Ny-1)
            return C_flux_bc;

        const csNumber_t& ci1 = C1(x, y+1);
        const csNumber_t& ci2 = C1(x, y-1);
        return (ci1 - ci2) / (2*dy);
    }
    csNumber_t dC2_dy(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 use the boundary value (C_flux_bc = 0.0 in this example).
        if(y == 0 || y == Ny-1)
            return C_flux_bc;

        const csNumber_t& ci1 = C2(x, y+1);
        const csNumber_t& ci2 = C2(x, y-1);
        return (ci1 - ci2) / (2*dy);
    }

    // Second order partial derivative per x.
    csNumber_t d2C1_dx2(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 return 0.0 (no diffusion through boundaries).
        if(x == 0 || x == Nx-1)
            return csNumber_t(0.0);

        const csNumber_t& ci1 = C1(x+1, y);
        const csNumber_t& ci  = C1(x,   y);
        const csNumber_t& ci2 = C1(x-1, y);
        return (ci1 - 2*ci + ci2) / (dx*dx);
    }
    csNumber_t d2C2_dx2(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 return 0.0 (no diffusion through boundaries).
        if(x == 0 || x == Nx-1)
            return csNumber_t(0.0);

        const csNumber_t& ci1 = C2(x+1, y);
        const csNumber_t& ci  = C2(x,   y);
        const csNumber_t& ci2 = C2(x-1, y);
        return (ci1 - 2*ci + ci2) / (dx*dx);
    }

    // Second order partial derivative per y.
    csNumber_t d2C1_dy2(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 return 0.0 (no diffusion through boundaries).
        if(y == 0 || y == Ny-1)
            return csNumber_t(0.0);

        const csNumber_t& ci1 = C1(x, y+1);
        const csNumber_t& ci  = C1(x,   y);
        const csNumber_t& ci2 = C1(x, y-1);
        return (ci1 - 2*ci + ci2) / (dy*dy);
    }
    csNumber_t d2C2_dy2(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 return 0.0 (no diffusion through boundaries).
        if(y == 0 || y == Ny-1)
            return csNumber_t(0.0);

        const csNumber_t& ci1 = C2(x, y+1);
        const csNumber_t& ci  = C2(x,   y);
        const csNumber_t& ci2 = C2(x, y-1);
        return (ci1 - 2*ci + ci2) / (dy*dy);
    }

public:
    int    Nequations;
    int    Nx;
    int    Ny;
    real_t V;
    real_t Kh;
    real_t Kv0;
    real_t q1;
    real_t q2;
    real_t C3;
    real_t a3;
    real_t a4;

protected:
    int    C1_start_index;
    int    C2_start_index;
    real_t x0, x1;
    real_t y0, y1;
    real_t dx;
    real_t dy;
    std::vector<real_t> x_domain;
    std::vector<real_t> y_domain;
    const csNumber_t* C1_values;
    const csNumber_t* C2_values;
    const csNumber_t& C_flux_bc;
};

#endif
