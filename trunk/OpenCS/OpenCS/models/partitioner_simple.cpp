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
#include <string>
#include <stdexcept>
#include "partitioner_simple.h"
#include "cs_partitioners.h"

namespace cs
{
std::shared_ptr<csGraphPartitioner_t> createGraphPartitioner_Simple()
{
    return std::shared_ptr<csGraphPartitioner_t>(new csGraphPartitioner_Simple);
}

csGraphPartitioner_Simple::csGraphPartitioner_Simple()
{
}

csGraphPartitioner_Simple::~csGraphPartitioner_Simple()
{
}

std::string csGraphPartitioner_Simple::GetName()
{
    return "Simple";
}

int csGraphPartitioner_Simple::Partition(int32_t                               Npe,
                                         int32_t                               Nvertices,
                                         int32_t                               Nconstraints,
                                         std::vector<uint32_t>&                rowIndices,
                                         std::vector<uint32_t>&                colIndices,
                                         std::vector< std::vector<int32_t> >&  vertexWeights,
                                         std::vector< std::set<int32_t> >&     partitions)
{
    int32_t eqCounter = 0;
    int32_t Nequations_PE = uint32_t(Nvertices/Npe);
    partitions.resize(Npe);
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        int32_t rangeStart = eqCounter;
        int32_t rangeEnd   = eqCounter + Nequations_PE;
        if(rangeEnd > Nvertices || pe == Npe-1)
            rangeEnd = Nvertices;

        std::set<int32_t>& partition = partitions[pe];
        for(int32_t ei = rangeStart; ei < rangeEnd; ei++)
           partition.insert(ei);

        eqCounter += Nequations_PE;
    }

    return 0;
}

}
