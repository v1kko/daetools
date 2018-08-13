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
#ifndef ROBERTS_MODEL_H
#define ROBERTS_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <math.h>
#include <OpenCS/models/cs_number.h>
using namespace cs;

class Roberts
{
public:
    Roberts()
    {
        Nequations = 3;
    }

    void SetInitialConditions(std::vector<real_t>& y0)
    {
        y0.assign(Nequations, 0.0);

        y0[0] = 1.0;
        y0[1] = 0.0;
        y0[2] = 0.0;
    }

    void CreateEquations(const std::vector<csNumber_t>& y_vars,
                         std::vector<csNumber_t>& equations)
    {
        if(y_vars.size() != Nequations)
            std::runtime_error("Invalid size of data arrays (must be 3)");

        equations.resize(Nequations);

        const csNumber_t& y1 = y_vars[0];
        const csNumber_t& y2 = y_vars[1];
        const csNumber_t& y3 = y_vars[2];

        equations[0] = -0.04 * y1 + 1.0e4 * y2 * y3;
        equations[1] =  0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * cs::pow(y2, 2);
        equations[2] =  3.0e7 * cs::pow(y2, 2);
    }

public:
    int Nequations;
};

#endif
