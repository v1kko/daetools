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
#include "../models/cs_model_builder.h"

/* Run using:
    $ mpirun -np 4 ./csSimulator_DAE inputDataDir
    $ mpirun -np 4 --hostfile mpi-hosts ./csSimulator_DAE inputDataDir
    $ mpirun -np 4 konsole --hold -e ./csSimulator_DAE inputDataDir
    $ mpirun -np 4 konsole --hold -e gdb -ex run --args ./csSimulator_DAE inputDataDir
*/
int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: csSimulator_DAE inputDataDir\n");
        return -1;
    }

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string inputDirectory = argv[1];
    cs::csSimulate_DAE(inputDirectory);

    MPI_Finalize();

    return 0;
}


