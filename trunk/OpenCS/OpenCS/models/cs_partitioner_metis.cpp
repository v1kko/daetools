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
#include "cs_partitioners.h"

// Metis also defines real_t, therefore disable daetools version.
#ifdef real_t
#undef real_t
#endif
#include <metis.h>

namespace cs
{
csGraphPartitioner_Metis::csGraphPartitioner_Metis(MetisRoutine routine)
{
    metisRoutine = routine;

    std::vector<idx_t> options_m(METIS_NOPTIONS, 0);
    METIS_SetDefaultOptions(&options_m[0]);
    metisOptions.insert(metisOptions.end(), options_m.begin(), options_m.end());
}

csGraphPartitioner_Metis::~csGraphPartitioner_Metis()
{
}

std::string csGraphPartitioner_Metis::GetName()
{
    if(metisRoutine == PartGraphKway)
        return "Metis_PartGraphKway";
    else if(metisRoutine == PartGraphRecursive)
        return "Metis_PartGraphRecursive";
    else
        return "Metis";
}

std::vector<int32_t> csGraphPartitioner_Metis::GetOptions() const
{
    return metisOptions;
}

void csGraphPartitioner_Metis::SetOptions(const std::vector<int32_t>& options)
{
    if(options.size() != METIS_NOPTIONS)
        throw std::runtime_error("Metis: Invalid size of the options array");
    metisOptions = options;
}

int csGraphPartitioner_Metis::Partition(int32_t                               Npe,
                                        int32_t                               Nvertices,
                                        int32_t                               Nconstraints,
                                        std::vector<uint32_t>&                rowIndices,
                                        std::vector<uint32_t>&                colIndices,
                                        std::vector< std::vector<int32_t> >&  vertexWeights,
                                        std::vector< std::set<int32_t> >&     partitions)
{
    // Check inputs
    if(vertexWeights.size() != Nconstraints)
        throw std::runtime_error("Metis: Invalid size of the vertexWeights array");
    for(uint32_t c = 0; c < Nconstraints; c++)
        if(vertexWeights[c].size() != Nvertices)
            throw std::runtime_error("Metis: Invalid size of the vertexWeights[" + std::to_string(c) + "] array");

    // metisOptions array is of type int32_t that can be different from idx_t (could be defined as int64_t).
    // Therefore, copy the options to a local std::vector<idx_t> array.
    std::vector<idx_t> options_v;
    options_v.insert(options_v.end(), metisOptions.begin(), metisOptions.end());

    // Important!!!
    // This is a critical phase for large systems and HAS to be optimised.
    // METIS requires that, for each edge between vertices v and u, both (v, u) and (u, v) to be stored.
    // Therefore, first construct an array of sets and fill-in the missing indexes.
    // Then, create two new arrays xadj_v and adjncy_v (CRS format) with the METIS adjacency structure of the graph.
    // Finally, free the memory held by the array of sets.
    std::vector<int32_t> xadj_v;
    std::vector<int32_t> adjncy_v;

    // Add the missing connections/edges.
    int Nnz = colIndices.size();
    std::vector< std::set<int32_t> > incidenceMatrix(Nvertices);
    for(int32_t vi = 0; vi < Nvertices; vi++)
    {
        std::set<int32_t>& v_set = incidenceMatrix[vi];
        for(uint32_t j = rowIndices[vi]; j < rowIndices[vi+1]; j++)
        {
            // Add ti to origin set
            int32_t ti = colIndices[j];
            v_set.insert(ti);

            // Add vi to target set
            std::set<int32_t>& t_set = incidenceMatrix[ti];
            t_set.insert(vi);
        }
    }

    // Reserve the memory.
    xadj_v.resize(Nvertices+1, 0);
    adjncy_v.reserve(2*Nnz); // this should be enough even in the worst case

    for(int32_t vi = 0; vi < Nvertices; vi++)
    {
        std::set<int32_t>& v_set = incidenceMatrix[vi];

        xadj_v[vi + 1] = xadj_v[vi] + v_set.size();
        adjncy_v.insert(adjncy_v.end(), v_set.begin(), v_set.end());
    }

    // Completely free the memory from the incidenceMatrix array.
    incidenceMatrix = std::vector< std::set<int32_t> >();

/*
    printf("IncidenceMatrix %d - %u\n", Nnz, adjncy_v.size());
    for(int32_t vi = 0; vi < Nvertices; vi++)
    {
        std::set<int32_t>& v_set = incidenceMatrix[vi];
        printf("  [%d]: [", vi);
        for(std::set<int32_t>::const_iterator it = v_set.begin(); it != v_set.end(); it++)
            printf("%d ", *it);
        printf("\n");
    }
    printf("adjncy_v:\n");
    for(std::vector<int32_t>::const_iterator it = adjncy_v.begin(); it != adjncy_v.end(); it++)
        printf("%d ", *it);
    printf("\n");
*/

    // Vertex weights in the flattened vweights[Nvertices,Nc] array format.
    std::vector<int32_t> vweights_flat;
    vweights_flat.reserve(Nvertices * Nconstraints);
    for(int32_t v = 0; v < Nvertices; v++)
        for(int32_t c = 0; c < Nconstraints; c++)
            vweights_flat.push_back(vertexWeights[c][v]);

    // Arrays to store the results
    std::vector<int32_t> part_v(Nvertices, 0);

    idx_t   nvtxs   = Nvertices;
    idx_t   ncon    = (Nconstraints == 0 ? 1 : Nconstraints);
    idx_t*  xadj    = &xadj_v[0];
    idx_t*  adjncy  = &adjncy_v[0];
    idx_t*  vwgt    = &vweights_flat[0];
    idx_t*  vsize   = NULL;
    idx_t*  adjwgt  = NULL;
    idx_t   nparts  = Npe;
    real_t* tpwgts  = NULL;
    real_t* ubvec   = NULL;
    idx_t*  options = (options_v.size() == METIS_NOPTIONS ? &options_v[0] : NULL);
    idx_t   edgecut = 0;
    idx_t*  part    = &part_v[0]; // Is part array allocated by METIS? It seems not...

    //if(metisRoutine == PartGraphKway)
    //    printf("PartGraphKway\n");
    //else if(metisRoutine == PartGraphRecursive)
    //    printf("PartGraphRecursive\n");

    int res;
    if(metisRoutine == PartGraphKway)
        res = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                                  &nparts, tpwgts, ubvec, options, &edgecut, part);
    else if(metisRoutine == PartGraphRecursive)
        res = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                                       &nparts, tpwgts, ubvec, options, &edgecut, part);

    if(res == METIS_OK)
    {
        partitions.resize(Npe);
        for(int32_t ei = 0; ei < Nvertices; ei++)
        {
            int32_t pe = part[ei];
            std::set<int32_t>& partition = partitions[pe];
            partition.insert(ei);
        }
    }
    else
    {
        std::string msg = "kMETIS partitioner failed: ";
        if(res == METIS_ERROR_INPUT)
            msg += "METIS_ERROR_INPUT";
        else if(res == METIS_ERROR_MEMORY)
            msg += "METIS_ERROR_MEMORY";
        else if(res == METIS_ERROR)
            msg += "METIS_ERROR";

        throw std::runtime_error(msg);
    }

    return (int)edgecut;
}

}
