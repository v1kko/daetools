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
#ifndef CHEMICAL_KINETICS_MODEL_H
#define CHEMICAL_KINETICS_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <math.h>
#include <OpenCS/models/cs_number.h>
using namespace cs;

class ChemicalKinetics
{
public:
    ChemicalKinetics()
    {
        Nequations = 6;

        k1   =  18.70;
        k2   =   0.58;
        k3   =   0.09;
        k4   =   0.42;
        K    =  34.40;
        klA  =   3.30;
        Ks   = 115.83;
        pCO2 =   0.90;
        H    = 737.00;
    }

    void SetInitialConditions(std::vector<real_t>& y0)
    {
        y0.assign(Nequations, 0.0);

        y0[0] = 0.444;
        y0[1] = 0.00123;
        y0[2] = 0.00;
        y0[3] = 0.007;
        y0[4] = 0.0;
        y0[5] = Ks * y0[0] * y0[3];
    }

    void GetVariableNames(std::vector<std::string>& names)
    {
        std::vector<std::string> vars = {"y1", "y2", "y3", "y4", "y5", "y6"};
        names = vars;
    }

    void CreateEquations(const std::vector<csNumber_t>& y,
                         const std::vector<csNumber_t>& dydt,
                         std::vector<csNumber_t>& equations)
    {
        if(y.size() != Nequations || dydt.size() != Nequations)
            std::runtime_error("Invalid size of data arrays (must be 6)");

        equations.resize(Nequations);

        const csNumber_t& y1 = y[0];
        const csNumber_t& y2 = y[1];
        const csNumber_t& y3 = y[2];
        const csNumber_t& y4 = y[3];
        const csNumber_t& y5 = y[4];
        const csNumber_t& y6 = y[5];

        const csNumber_t& dy1_dt = dydt[0];
        const csNumber_t& dy2_dt = dydt[1];
        const csNumber_t& dy3_dt = dydt[2];
        const csNumber_t& dy4_dt = dydt[3];
        const csNumber_t& dy5_dt = dydt[4];

        csNumber_t r1  = k1 * cs::pow(y1,4) * cs::sqrt(y2);
        csNumber_t r2  = k2 * y3 * y4;
        csNumber_t r3  = k2/K * y1 * y5;
        csNumber_t r4  = k3 * y1 * y4 * y4;
        csNumber_t r5  = k4 * y6 * y6 * cs::sqrt(y2);
        csNumber_t Fin = klA * ( pCO2/H - y2 );

        equations[0] = dy1_dt + 2*r1 - r2 + r3 + r4;
        equations[1] = dy2_dt + 0.5*r1 + r4 + 0.5*r5 - Fin;
        equations[2] = dy3_dt - r1 + r2 - r3;
        equations[3] = dy4_dt + r2 - r3 + 2*r4;
        equations[4] = dy5_dt - r2 + r3 - r5;
        equations[5] = Ks*y1*y4 - y6;
    }

public:
    int    Nequations;
    real_t k1;
    real_t k2;
    real_t k3;
    real_t k4;
    real_t K;
    real_t klA;
    real_t Ks;
    real_t pCO2;
    real_t H;
};

#endif
