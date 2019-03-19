/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with the
OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef ADVECTION_DIFFUSION_MODEL_H
#define ADVECTION_DIFFUSION_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <math.h>
#include <OpenCS/models/cs_number.h>
using namespace cs;

class AdvectionDiffusion_2D
{
public:
    AdvectionDiffusion_2D(int nx, int ny, const csNumber_t& bc):
        Nx(nx), Ny(ny), u_bc(bc)
    {
        // In the CVode example cvsAdvDiff_bnd.c they only modelled interior points,
        //   excluded the boundaries from the ODE system, and assumed homogenous Dirichlet BCs (0.0).
        // There, they divided the 2D domain into (Nx+1) by (Ny+1) points and
        //   the points at x=0, x=Lx, y=0 and y=Ly are not used in the model.
        // Thus, x domain starts at x=1*dx, y domain starts at x=1*dy.
        x0 = 0.0;
        x1 = 2.0;
        y0 = 0.0;
        y1 = 1.0;
        dx = (x1-x0) / (Nx+2-1);
        dy = (y1-y0) / (Ny+2-1);

        Nequations = Nx*Ny;
        u_values   = NULL;
    }

    void SetInitialConditions(std::vector<real_t>& u0)
    {
        int ix, iy;
        u0.assign(Nequations, 0.0);

        for(ix = 0; ix < Nx; ix++)
        {
            for(iy = 0; iy < Ny; iy++)
            {
                int index = getIndex(ix,iy);

                real_t x = (ix+1) * dx;
                real_t y = (iy+1) * dy;

                u0[index] = x*(x1 - x)*y*(y1 - y)*std::exp(5*x*y);
            }
        }
    }

    void GetVariableNames(std::vector<std::string>& names)
    {
        const int bsize = 32;
        char buffer[bsize];
        int index = 0;

        names.resize(Nequations);
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                std::snprintf(buffer, bsize, "%s(%d,%d)", "u", x, y);
                names[index] = buffer;
                index++;
            }
        }
    }

    void CreateEquations(const std::vector<csNumber_t>& u_vars,
                         std::vector<csNumber_t>& equations)
    {
        if(u_vars.size() != Nequations)
            std::runtime_error("Invalid size of data arrays");

        equations.resize(Nequations);

        u_values = &u_vars[0];

        int eq = 0;
        for(int x = 0; x < Nx; x++)
            for(int y = 0; y < Ny; y++)
                equations[eq++] = d2u_dx2(x,y) + 0.5 * du_dx(x,y) + d2u_dy2(x,y);
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

    const csNumber_t& u(int x, int y)
    {
        int index = getIndex(x, y);
        return u_values[index];
    }

    // First order partial derivative per x.
    csNumber_t du_dx(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 use the boundary value (u_bc = 0.0 in this example).
        const csNumber_t& ui1 = (x == Nx-1 ? u_bc : u(x+1, y));
        const csNumber_t& ui2 = (x == 0    ? u_bc : u(x-1, y));
        return (ui1 - ui2) / (2*dx);
    }

    // First order partial derivative per y.
    // Not used in this example.
    csNumber_t du_dy(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 use the boundary value (u_bc = 0.0 in this example).
        const csNumber_t& ui1 = (y == Ny-1 ? u_bc : u(x, y+1));
        const csNumber_t& ui2 = (y == 0    ? u_bc : u(x, y-1));
        return (ui1 - ui2) / (2*dy);
    }

    // Second order partial derivative per x.
    csNumber_t d2u_dx2(int x, int y)
    {
        // If called for x == 0 or x == Nx-1 use the boundary value (u_bc = 0.0 in this example).
        const csNumber_t& ui1 = (x == Nx-1 ? u_bc : u(x+1, y));
        const csNumber_t& ui  =                     u(x,   y);
        const csNumber_t& ui2 = (x == 0    ? u_bc : u(x-1, y));
        return (ui1 - 2*ui + ui2) / (dx*dx);
    }

    // Second order partial derivative per y.
    csNumber_t d2u_dy2(int x, int y)
    {
        // If called for y == 0 or y == Ny-1 use the boundary value (u_bc = 0.0 in this example).
        const csNumber_t& ui1 = (y == Ny-1 ? u_bc : u(x, y+1));
        const csNumber_t& ui  =                     u(x,   y);
        const csNumber_t& ui2 = (y == 0    ? u_bc : u(x, y-1));
        return (ui1 - 2*ui + ui2) / (dy*dy);
    }

public:
    int Nequations;
    int Nx;
    int Ny;

protected:
    real_t x0, x1;
    real_t y0, y1;
    real_t dx;
    real_t dy;
    const csNumber_t* u_values;
    const csNumber_t& u_bc;
};

#endif
