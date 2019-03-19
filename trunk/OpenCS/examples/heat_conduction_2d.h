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
#ifndef HEAT_CONDUCTION_MODEL_H
#define HEAT_CONDUCTION_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <OpenCS/models/cs_number.h>
using namespace cs;

// DAE Tools tutorial1
class HeatConduction_2D
{
public:
    HeatConduction_2D(int nx, int ny):
        Nx(nx), Ny(ny)
    {
        real_t Lx = 0.1; // m
        real_t Ly = 0.1; // m

        dx = Lx / (Nx-1);
        dy = Ly / (Ny-1);

        Nequations = Nx*Ny;
        T_values = NULL;
        T_derivs = NULL;

        rho = 8960; // density, kg/m^3
        cp  =  385; // specific heat capacity, J/(kg.K)
        k   =  401; // thermal conductivity, W/(m.K)
        Qb  =  1E5; // flux at the bottom edge, W/m^2
        Tt  =  300; // T at the top edge, K
    }

    void SetInitialConditions(std::vector<real_t>& T0)
    {
        T0.assign(Nequations, 300.0);
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
                std::snprintf(buffer, bsize, "%s(%d,%d)", "T", x, y);
                names[index] = buffer;
                index++;
            }
        }
    }

    void CreateEquations(const std::vector<csNumber_t>& T_vars,
                         const std::vector<csNumber_t>& dTdt_vars,
                         std::vector<csNumber_t>& equations)
    {
        T_values = &T_vars[0];
        T_derivs = &dTdt_vars[0];

        equations.resize(Nequations);

        int eq = 0;
        for(int x = 0; x < Nx; x++)
        {
            for(int y = 0; y < Ny; y++)
            {
                if(x == 0) // Left BC: zero flux
                {
                    equations[eq++] = dT_dx(x,y);
                }
                else if(x == Nx-1) // Right BC: zero flux
                {
                    equations[eq++] = dT_dx(x,y);
                }
                else if(x > 0 && x < Nx-1 && y == 0) // Bottom BC: prescribed flux
                {
                    equations[eq++] = -k * dT_dy(x,y) - Qb;
                }
                else if(x > 0 && x < Nx-1 && y == Ny-1) // Top BC: prescribed flux
                {
                    equations[eq++] = T(x,y) - Tt;
                }
                else // Inner region: diffusion equation
                {
                    equations[eq++] = rho * cp * dT_dt(x,y) - k * (d2T_dx2(x,y) + d2T_dy2(x,y));
                }
            }
        }
    }

protected:
    int GetIndex(int x, int y)
    {
        if(x < 0 || x >= Nx)
            throw std::runtime_error("Invalid x index");
        if(y < 0 || y >= Ny)
            std::runtime_error("Invalid y index");
        return Ny*x + y;
    }

    const csNumber_t& T(int x, int y)
    {
        int index = Ny*x + y;
        return T_values[index];
    }

    const csNumber_t& dT_dt(int x, int y)
    {
        int index = Ny*x + y;
        return T_derivs[index];
    }

    // First order partial derivative per x.
    csNumber_t dT_dx(int x, int y)
    {
        if(x == 0) // left
        {
            const csNumber_t& T0 = T(0, y);
            const csNumber_t& T1 = T(1, y);
            const csNumber_t& T2 = T(2, y);
            return (-3*T0 + 4*T1 - T2) / (2*dx);
        }
        else if(x == Nx-1) // right
        {
            const csNumber_t& Tn  = T(Nx-1,   y);
            const csNumber_t& Tn1 = T(Nx-1-1, y);
            const csNumber_t& Tn2 = T(Nx-1-2, y);
            return (3*Tn - 4*Tn1 + Tn2) / (2*dx);
        }
        else
        {
            const csNumber_t& T1 = T(x+1, y);
            const csNumber_t& T2 = T(x-1, y);
            return (T1 - T2) / (2*dx);
        }
    }

    // First order partial derivative per y.
    csNumber_t dT_dy(int x, int y)
    {
        if(y == 0) // bottom
        {
            const csNumber_t& T0 = T(x, 0);
            const csNumber_t& T1 = T(x, 1);
            const csNumber_t& T2 = T(x, 2);
            return (-3*T0 + 4*T1 - T2) / (2*dy);
        }
        else if(y == Ny-1) // top
        {
            const csNumber_t& Tn  = T(x, Ny-1  );
            const csNumber_t& Tn1 = T(x, Ny-1-1);
            const csNumber_t& Tn2 = T(x, Ny-1-2);
            return (3*Tn - 4*Tn1 + Tn2) / (2*dy);
        }
        else
        {
            const csNumber_t& Ti1 = T(x, y+1);
            const csNumber_t& Ti2 = T(x, y-1);
            return (Ti1 - Ti2) / (2*dy);
        }
    }

    // Second order partial derivative per x.
    csNumber_t d2T_dx2(int x, int y)
    {
        // This function is typically called only for interior points.
        if(x == 0 || x == Nx-1)
            throw std::runtime_error("d2T_dx2 called for boundary x point");

        const csNumber_t& Ti1 = T(x+1, y);
        const csNumber_t& Ti  = T(x,   y);
        const csNumber_t& Ti2 = T(x-1, y);
        return (Ti1 - 2*Ti + Ti2) / (dx*dx);
    }

    // Second order partial derivative per y.
    csNumber_t d2T_dy2(int x, int y)
    {
        // This function is typically called only for interior points.
        if(y == 0 || y == Ny-1)
            throw std::runtime_error("d2T_dy2 called for boundary y point");

        const csNumber_t& Ti1 = T(x, y+1);
        const csNumber_t& Ti  = T(x,   y);
        const csNumber_t& Ti2 = T(x, y-1);
        return (Ti1 - 2*Ti + Ti2) / (dy*dy);
    }

public:
    int    Nequations;
    int    Nx;
    int    Ny;
    real_t rho;
    real_t cp;
    real_t k;
    real_t Qb;
    real_t Tt;

protected:
    real_t dx;
    real_t dy;
    const csNumber_t* T_values;
    const csNumber_t* T_derivs;
};

#endif
