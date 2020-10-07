/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_GRAPH_PARTITIONERS_H
#define CS_GRAPH_PARTITIONERS_H

#include <stdint.h>
#include <set>
#include <vector>
#include <string>
#include "../cs_model.h"

namespace cs
{
OPENCS_MODELS_API std::shared_ptr<csGraphPartitioner_t> createGraphPartitioner_Simple();
OPENCS_MODELS_API std::shared_ptr<csGraphPartitioner_t> createGraphPartitioner_Metis(const std::string& algorithm);
OPENCS_MODELS_API std::shared_ptr<csGraphPartitioner_t> createGraphPartitioner_2D_Npde(int N_x, int N_y, int N_pde, double Npex_Npey_ratio = 1.0);
}

#endif
