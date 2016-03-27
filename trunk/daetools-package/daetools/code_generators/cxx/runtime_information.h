/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2016
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_RUNTIME_INFORMATION_H
#define DAE_RUNTIME_INFORMATION_H

#include <string>
#include <vector>
#include <map>
using namespace std;

struct runtimeInformationData
{
    int            i_start;
    int            i_end;
    vector<real_t> init_values;
    vector<real_t> init_derivatives;
    vector<real_t> absolute_tolerances;
    vector<int>    ids;
    vector<string> variable_names;
};

%(runtimeInformation_h)s

#endif
