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
#ifndef BRUSSELATOR_MODEL_H
#define BRUSSELATOR_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <math.h>
#include <OpenCS/models/cs_number.h>
using namespace cs;

class Brusselator_2D
{
public:
    Brusselator_2D(int nx, int ny, const csNumber_t& bc_u_flux, const csNumber_t& bc_v_flux):
        Nx(nx), Ny(ny), u_flux_bc(bc_u_flux), v_flux_bc(bc_v_flux)
    {
        Nequations = 2*Nx*Ny;

        x0 = 0.0;
        x1 = 1.0;
        y0 = 0.0;
        y1 = 1.0;
        dx = (x1-x0) / (Nx-1);
        dy = (y1-y0) / (Ny-1);

        x_domain.resize(Nx);
        y_domain.resize(Ny);
        for(int x = 0; x < Nx; x++)
            x_domain[x] = x0 + x * dx;
        for(int y = 0; y < Ny; y++)
            y_domain[y] = y0 + y * dy;

        u_start_index = 0*Nx*Ny;
        v_start_index = 1*Nx*Ny;

        eps1 = 0.002;
        eps2 = 0.002;
        A    = 1.000;
        B    = 3.000;

        u_data     = NULL;
        u_data     = NULL;
        du_dt_data = NULL;
        dv_dt_data = NULL;
    }

    void SetInitialConditions(std::vector<real_t>& uv0, std::vector<real_t>& uv0_dt)
    {
        int ix, iy;
        uv0.assign   (Nequations, 0.0);
        uv0_dt.assign(Nequations, 0.0);

        real_t pi = 3.1415926535898;

        real_t* u_0 = &uv0[u_start_index];
        real_t* v_0 = &uv0[v_start_index];

        for(ix = 0; ix < Nx; ix++)
        {
            for(iy = 0; iy < Ny; iy++)
            {
                int index = getIndex(ix,iy);

                real_t x = x_domain[ix];
                real_t y = y_domain[iy];

                u_0[index] = 1.0 - 0.5 * std::cos(pi*y);
                v_0[index] = 3.5 - 2.5 * std::cos(pi*x);
            }
        }
    }

    void CreateEquations(const std::vector<csNumber_t>& values,
                         const std::vector<csNumber_t>& derivs,
                         std::vector<csNumber_t>& equations)
    {
        u_data     = &values[u_start_index];
        v_data     = &values[v_start_index];
        du_dt_data = &derivs[u_start_index];
        dv_dt_data = &derivs[v_start_index];

        equations.resize(Nequations);

        int eq = 0;
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                /* u component */
                if(x == 0)          // Left BC: Neumann BCs
                {
                    equations[eq++] = du_dx(x,y) - u_flux_bc;
                }
                else if(x == Nx-1)  // Right BC: Neumann BCs
                {
                    equations[eq++] = du_dx(x,y) - u_flux_bc;
                }
                else if(y == 0)     // Bottom BC: Neumann BCs
                {
                    equations[eq++] = du_dy(x,y) - u_flux_bc;
                }
                else if(y == Ny-1)  // Top BC: Neumann BCs
                {
                    equations[eq++] = du_dy(x,y) - u_flux_bc;
                }
                else
                {
                    // Interior points
                    equations[eq++]  = du_dt(x,y) -                              /* accumulation term */
                                       (
                                         eps1 * (d2u_dx2(x,y) + d2u_dy2(x,y)) +  /* diffusion term    */
                                         u(x,y)*u(x,y)*v(x,y) - (B+1)*u(x,y) + A /* generation term   */
                                       );
                }
            }
        }
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                /* v component */
                if(x == 0)          // Left BC: Neumann BCs
                {
                    equations[eq++] = dv_dx(x,y) - v_flux_bc;
                }
                else if(x == Nx-1)  // Right BC: Neumann BCs
                {
                    equations[eq++] = dv_dx(x,y) - v_flux_bc;
                }
                else if(y == 0)     // Bottom BC: Neumann BCs
                {
                    equations[eq++] = dv_dy(x,y) - v_flux_bc;
                }
                else if(y == Ny-1)  // Top BC: Neumann BCs
                {
                    equations[eq++] = dv_dy(x,y) - v_flux_bc;
                }
                else
                {
                    // Interior points
                    equations[eq++]  = dv_dt(x,y) -                             /* accumulation term */
                                       (
                                         eps2 * (d2v_dx2(x,y) + d2v_dy2(x,y)) - /* diffusion term    */
                                         u(x,y)*u(x,y)*v(x,y) + B*u(x,y)        /* generation term   */
                                       );
                }
            }
        }
    }

protected:
    csNumber_t u(int x, int y)
    {
        int index = getIndex(x,y);
        return u_data[index];
    }
    csNumber_t v(int x, int y)
    {
        int index = getIndex(x,y);
        return v_data[index];
    }
    csNumber_t du_dt(int x, int y)
    {
        int index = getIndex(x,y);
        return du_dt_data[index];
    }
    csNumber_t dv_dt(int x, int y)
    {
        int index = getIndex(x,y);
        return dv_dt_data[index];
    }

    // First order partial derivative per x.
    csNumber_t du_dx(int x, int y)
    {
        if(x == 0) // left
        {
            const csNumber_t& u0 = u(0, y);
            const csNumber_t& u1 = u(1, y);
            const csNumber_t& u2 = u(2, y);
            return (-3*u0 + 4*u1 - u2) / (2*dx);
        }
        else if(x == Nx-1) // right
        {
            const csNumber_t& un  = u(Nx-1,   y);
            const csNumber_t& un1 = u(Nx-1-1, y);
            const csNumber_t& un2 = u(Nx-1-2, y);
            return (3*un - 4*un1 + un2) / (2*dx);
        }
        else
        {
            const csNumber_t& u1 = u(x+1, y);
            const csNumber_t& u2 = u(x-1, y);
            return (u1 - u2) / (2*dx);
        }
    }
    csNumber_t dv_dx(int x, int y)
    {
        if(x == 0) // left
        {
            const csNumber_t& u0 = v(0, y);
            const csNumber_t& u1 = v(1, y);
            const csNumber_t& u2 = v(2, y);
            return (-3*u0 + 4*u1 - u2) / (2*dx);
        }
        else if(x == Nx-1) // right
        {
            const csNumber_t& un  = v(Nx-1,   y);
            const csNumber_t& un1 = v(Nx-1-1, y);
            const csNumber_t& un2 = v(Nx-1-2, y);
            return (3*un - 4*un1 + un2) / (2*dx);
        }
        else
        {
            const csNumber_t& u1 = v(x+1, y);
            const csNumber_t& u2 = v(x-1, y);
            return (u1 - u2) / (2*dx);
        }
    }

    // First order partial derivative per y.
    csNumber_t du_dy(int x, int y)
    {
        if(y == 0) // bottom
        {
            const csNumber_t& u0 = u(x, 0);
            const csNumber_t& u1 = u(x, 1);
            const csNumber_t& u2 = u(x, 2);
            return (-3*u0 + 4*u1 - u2) / (2*dy);
        }
        else if(y == Ny-1) // top
        {
            const csNumber_t& un  = u(x, Ny-1  );
            const csNumber_t& un1 = u(x, Ny-1-1);
            const csNumber_t& un2 = u(x, Ny-1-2);
            return (3*un - 4*un1 + un2) / (2*dy);
        }
        else
        {
            const csNumber_t& ui1 = u(x, y+1);
            const csNumber_t& ui2 = u(x, y-1);
            return (ui1 - ui2) / (2*dy);
        }
    }
    csNumber_t dv_dy(int x, int y)
    {
        if(y == 0) // bottom
        {
            const csNumber_t& u0 = v(x, 0);
            const csNumber_t& u1 = v(x, 1);
            const csNumber_t& u2 = v(x, 2);
            return (-3*u0 + 4*u1 - u2) / (2*dy);
        }
        else if(y == Ny-1) // top
        {
            const csNumber_t& un  = v(x, Ny-1  );
            const csNumber_t& un1 = v(x, Ny-1-1);
            const csNumber_t& un2 = v(x, Ny-1-2);
            return (3*un - 4*un1 + un2) / (2*dy);
        }
        else
        {
            const csNumber_t& ui1 = v(x, y+1);
            const csNumber_t& ui2 = v(x, y-1);
            return (ui1 - ui2) / (2*dy);
        }
    }

    // Second order partial derivative per x.
    csNumber_t d2u_dx2(int x, int y)
    {
        if(x == 0 || x == Nx-1)
            std::runtime_error("d2u_dx2 called at the boundary");

        const csNumber_t& ui1 = u(x+1, y);
        const csNumber_t& ui  = u(x,   y);
        const csNumber_t& ui2 = u(x-1, y);
        return (ui1 - 2*ui + ui2) / (dx*dx);
    }
    csNumber_t d2v_dx2(int x, int y)
    {
        if(x == 0 || x == Nx-1)
            std::runtime_error("d2v_dx2 called at the boundary");

        const csNumber_t& vi1 = v(x+1, y);
        const csNumber_t& vi  = v(x,   y);
        const csNumber_t& vi2 = v(x-1, y);
        return (vi1 - 2*vi + vi2) / (dx*dx);
    }

    // Second order partial derivative per y.
    csNumber_t d2u_dy2(int x, int y)
    {
        if(y == 0 || y == Ny-1)
            std::runtime_error("d2u_dy2 called at the boundary");

        const csNumber_t& ui1 = u(x, y+1);
        const csNumber_t& ui  = u(x,   y);
        const csNumber_t& ui2 = u(x, y-1);
        return (ui1 - 2*ui + ui2) / (dy*dy);
    }
    csNumber_t d2v_dy2(int x, int y)
    {
        if(y == 0 || y == Ny-1)
            std::runtime_error("d2v_dy2 called at the boundary");

        const csNumber_t& vi1 = v(x, y+1);
        const csNumber_t& vi  = v(x,   y);
        const csNumber_t& vi2 = v(x, y-1);
        return (vi1 - 2*vi + vi2) / (dy*dy);
    }

    int getIndex(int x, int y)
    {
        if(x < 0 || x >= Nx)
            throw std::runtime_error("Invalid x index");
        if(y < 0 || y >= Ny)
            std::runtime_error("Invalid y index");
        return Ny*x + y;
    }

public:
    int    Nequations;
    int    Nx;
    int    Ny;
    real_t eps1;
    real_t eps2;
    real_t A;
    real_t B;

protected:
    int    u_start_index;
    int    v_start_index;
    real_t x0, x1;
    real_t y0, y1;
    real_t dx;
    real_t dy;
    std::vector<real_t> x_domain;
    std::vector<real_t> y_domain;
    const csNumber_t* u_data;
    const csNumber_t* v_data;
    const csNumber_t* du_dt_data;
    const csNumber_t* dv_dt_data;
    const csNumber_t& u_flux_bc;
    const csNumber_t& v_flux_bc;
};

#endif
