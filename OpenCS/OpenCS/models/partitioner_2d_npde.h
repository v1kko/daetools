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
#ifndef CS_GRAPH_PARTITIONER_2D_NPDE_H
#define CS_GRAPH_PARTITIONER_2D_NPDE_H

#include <stdint.h>
#include <set>
#include <vector>
#include <string>
#include "../cs_model.h"

namespace cs
{
/* Graph partitioner for Npde equations distributed on a uniform 2D grid (Nx by Ny).
 * Divides the 2D grid into Npe regions (at the moment only 4, 8 and 16 PEs).
 * The system contains Npde equations.
 * Variable indexes are consecutive:
 *   var1: [   0, N1]
 *   var2: [  N1, N2]
 *   ...
 *   varN: [Nn-1, Nn]
 * and equation indexes, too:
 *   pde1: [   0, N1]
 *   pde2: [  N1, N2]
 *   ...
 *   pdeN: [Nn-1, Nn]
 */
class OPENCS_MODELS_API csGraphPartitioner_2D_Npde: public csGraphPartitioner_t
{
public:
    typedef std::pair<int32_t,int32_t>   range_2D;
    typedef std::pair<range_2D,range_2D> peRanges_2D;
    typedef std::vector<peRanges_2D>     systemRanges_2D;

    /* Arguments:
     *  N_x:               Number of points along x axis
     *  N_y:               Number of points along y axis
     *  N_pde:             Number of PDEs in the system
     *  Npe_x_Npe_y_ratio: Ratio between Npe per x and Npe per y axis: Npe_x : Npe_y = Npe_x_Npe_y_scale
     *                     Used to set different number of PEs for x and y axes.
     *                     I.e. if Npe=128 and Npe_x_Npe_y_scale=0.5 -> 8x16 partitions. */
    csGraphPartitioner_2D_Npde(int N_x, int N_y, int N_pde, double Npex_Npey_ratio = 1.0);
    virtual ~csGraphPartitioner_2D_Npde();

    virtual std::string GetName();
    virtual int Partition(int32_t                               Npe,
                          int32_t                               Nvertices,
                          int32_t                               Nconstraints,
                          std::vector<uint32_t>&                rowIndices,
                          std::vector<uint32_t>&                colIndices,
                          std::vector< std::vector<int32_t> >&  vertexWeights,
                          std::vector< std::set<int32_t> >&     partitions);

    int Nx;
    int Ny;
    int Npde;
    int Nequations;
    double Npe_x_Npe_y_ratio;
};

}

#endif
