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
#include <cmath>
#include <stdexcept>
#include "partitioner_2d_npde.h"
#include "cs_partitioners.h"

namespace cs
{
std::shared_ptr<csGraphPartitioner_t> createGraphPartitioner_2D_Npde(int N_x, int N_y, int N_pde, double Npex_Npey_ratio)
{
    return std::shared_ptr<csGraphPartitioner_t>(new csGraphPartitioner_2D_Npde(N_x, N_y, N_pde, Npex_Npey_ratio));
}

csGraphPartitioner_2D_Npde::csGraphPartitioner_2D_Npde(int N_x, int N_y, int N_pde, double Npex_Npey_ratio)
{
    Nx                = N_x;
    Ny                = N_y;
    Npde              = N_pde;
    Npe_x_Npe_y_ratio = Npex_Npey_ratio;
}

csGraphPartitioner_2D_Npde::~csGraphPartitioner_2D_Npde()
{
}

std::string csGraphPartitioner_2D_Npde::GetName()
{
    return "2D-Npde";
}

int csGraphPartitioner_2D_Npde::Partition(int32_t                               Npe,
                                          int32_t                               Nvertices,
                                          int32_t                               Nconstraints,
                                          std::vector<uint32_t>&                rowIndices,
                                          std::vector<uint32_t>&                colIndices,
                                          std::vector< std::vector<int32_t> >&  vertexWeights,
                                          std::vector< std::set<int32_t> >&     partitions)
{
    if(Npde*Nx*Ny != Nvertices)
        csThrowException("Nvertices does not correspond to the number of equations: Npde*Nx*Ny");

    std::vector<int32_t> pde_start_indexes;
    for(int i = 0; i < Npde; i++)
        pde_start_indexes.push_back(i*Nx*Ny);

    // Npe_x_Npe_y_ratio is a ratio between Npe per x and Npe per y axis: Npe_x : Npe_y = Npe_x_Npe_y_scale
    double Nxy = std::sqrt(double(Npe) / Npe_x_Npe_y_ratio);
    int Npe_x = std::round(Nxy * Npe_x_Npe_y_ratio);
    int Npe_y = std::round(Nxy * 1.0);

    // If Npe_x*Npe_y does not match Npe - raise an exception
    if(Npe_x*Npe_y != Npe)
        csThrowException(std::string("Invalid Npex_Npey_ratio: Npe_x*Npe_y != Npe (") + std::to_string(Npe_x) + "*" + std::to_string(Npe_y) + "!=" + std::to_string(Npe) + ")");

    int Nx_per_pe = std::round(Nx / Npe_x);
    int Ny_per_pe = std::round(Ny / Npe_y);

    //printf("  %d, %d, %d, %d\n", Npe_x, Npe_y, Nx_per_pe, Ny_per_pe);

    printf("%s partitioner (Npe=%d, Nvertices=%d; Nx=%d, Ny=%d, Npde=%d, (Npe_x/Npe_y)-ratio=%f):\n",
           GetName().c_str(), Npe, Nvertices, Nx, Ny, Npde, Npe_x_Npe_y_ratio);
    printf("  No. partitions along x axis             = %d\n", Npe_x);
    printf("  No. partitions along y axis             = %d\n", Npe_y);
    printf("  No. points along x axis (per partition) = %d\n", Nx_per_pe);
    printf("  No. points along y axis (per partition) = %d\n", Ny_per_pe);
    if(Nx_per_pe*Npe_x != Nx)
    {
        printf("  WARNING:\n");
        printf("    The number of points in partitions along x axis is not uniform.\n");
        printf("    The last partition on x axis will get %d additional points\n", Nx - Nx_per_pe*Npe_x);
    }
    if(Ny_per_pe*Npe_y != Ny)
    {
        printf("  WARNING:\n");
        printf("    The number of points in partitions along y axis is not uniform.\n");
        printf("    The last partition on y axis will get %d additional points\n", Ny - Ny_per_pe*Npe_y);
    }
    printf("\n");

    int32_t pe = 0;
    partitions.resize(Npe);
    for(int32_t pey = 0; pey < Npe_y; pey++)
    {
        for(int32_t pex = 0; pex < Npe_x; pex++)
        {
            int32_t Nx_start = (pex  ) * Nx_per_pe;
            int32_t Nx_end   = (pex+1) * Nx_per_pe;
            int32_t Ny_start = (pey  ) * Ny_per_pe;
            int32_t Ny_end   = (pey+1) * Ny_per_pe;

            // Last PEs always get the remaining points
            if(pex == Npe_x-1 && Nx_end != Nx)
                Nx_end = Nx;
            if(pey == Npe_y-1 && Ny_end != Ny)
                Ny_end = Ny;

            // Total number of points in this PE
            int32_t Npe_total = (Nx_end-Nx_start) * (Ny_end-Ny_start);

            printf("  PE %d: [%5d,%5d) x [%5d,%5d) = %d unknowns\n", pe, Nx_start, Nx_end, Ny_start, Ny_end, Npe_total);

            for(int32_t x = Nx_start; x < Nx_end; x++)
            {
                for(int32_t y = Ny_start; y < Ny_end; y++)
                {
                    for(int32_t pde = 0; pde < Npde; pde++)
                    {
                        int32_t pde_start = pde_start_indexes[pde];
                        int32_t xy_index  = Ny*x + y;
                        partitions[pe].insert(pde_start + xy_index);
                    }
                }
            }
            pe++;
        }
    }
    return 0;
}

}
