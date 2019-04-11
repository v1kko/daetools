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
#include <mpi.h>
#include "cs_simulators.h"

/* Single CPU simulations:
    $ ./csSimulator inputDataDir

   MPI simulations (OpenMPI):
    $ mpirun -np Npe                      ./csSimulator inputDataDir
    $ mpirun -np Npe --hostfile mpi-hosts ./csSimulator inputDataDir
   or (MSMPI and MPICH):
    $ mpiexec -n Npe                      ./csSimulator inputDataDir
    $ mpiexec -n Npe --hostfile mpi-hosts ./csSimulator inputDataDir

   GDB debugging:
    $ mpirun -np Npe gdb -ex run --args ./csSimulator inputDataDir
*/
int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: csSimulator inputDataDir\n");
        return -1;
    }

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string inputDirectory = argv[1];
    cs::csSimulate(inputDirectory);

    MPI_Finalize();

    return 0;
}


