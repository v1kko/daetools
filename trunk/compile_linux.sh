#!/bin/sh

set -e

if [ "$1" = "" ] || [ "$1" = "--help" ] || [ "$1" = "-help" ] || [ "$1" = "-h" ]; then
  echo "Usage: compile_linux [commands]"
  echo "Commands: all | core | pydae | trilinos | superlu | superlu_mt | superlu_cuda | bonmin | ipopt | nlopt"
  echo "  - all: build all daetools libraries, solvers and python wrapper modules"
  echo "  - core: build core daetools c++ libraries and python wrapper modules (no 3rd party LA/(MI)NLP solvers)"
  echo "  - pydae: build daetools core python wrapper modules only"
  echo "  - solvers: build all solvers and their python wrapper modules"
  echo "  - trilinos, superlu, superlu_mt, superlu_cuda, bonmin, ipopt, nlopt, intel_pardiso: build particular solver and its python wrapper module"
  return
fi

PROJECTS=$1

HOST_ARCH=`uname -m`
PLATFORM=`uname -s`
TRUNK="$( cd "$( dirname "$0" )" && pwd )"

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  SPEC=macx-g++

else
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
  SPEC=linux-g++
fi

if [ ${Ncpu} -gt 1 ]; then
  Ncpu=$(($Ncpu+1))
fi
#echo "Number of threads: ${Ncpu}"

cd ${TRUNK}

if [ ! -d release ]; then
    mkdir release
fi

compile () {
  DIR=$1
  MAKEARG=$2
  CONFIG=$3
  echo ""
  echo "****************************************************************"
  echo "                Build the project: $DIR"
  echo "****************************************************************"
  
  if [ ${DIR} = "dae" ]; then
    cd ${TRUNK}
  elif [ ${DIR} = "pyCore" ]; then
    cd ${TRUNK}/pyCore
  elif [ ${DIR} = "pyActivity" ]; then
    cd ${TRUNK}/pyActivity
  elif [ ${DIR} = "pyDataReporting" ]; then
    cd ${TRUNK}/pyDataReporting
  elif [ ${DIR} = "pyIDAS" ]; then
    cd ${TRUNK}/pyIDAS
  elif [ ${DIR} = "pyUnits" ]; then
    cd ${TRUNK}/pyUnits
  elif [ ${DIR} = "pySuperLU" ]; then
    cd ${TRUNK}/pySuperLU
  elif [ ${DIR} = "pyBonmin" ]; then
    cd ${TRUNK}/pyBonmin
  elif [ ${DIR} = "pyTrilinos" ]; then
    cd ${TRUNK}/pyTrilinos
  else
    cd ${DIR}
  fi

  echo ""
  echo "[*] Configuring the project with ($2, $3)..."
  echo ""
  qmake -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile -spec ${SPEC} ${CONFIG}
  
  echo ""
  echo "[*] Cleaning the project..."
  echo ""
  make clean -w
  
  echo ""
  echo "[*] Compiling the project..."
  echo ""
  make ${MAKEARG} -w
  
  echo ""
  echo "[*] Done!"
  cd ${TRUNK}
}

case ${PROJECTS} in
  all)  echo [all]
        cd ${TRUNK}/release
        rm -rf *
        cd ${TRUNK}

        compile dae                "-j$Ncpu"

        compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU"
        compile pySuperLU          "-j1" "CONFIG+=shellSuperLU"

        compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU_MT"
        compile pySuperLU          "-j1" "CONFIG+=shellSuperLU_MT"

        #compile LA_SUPERLU         "-j1 --file=gpuMakefile"  "CONFIG+=shellSuperLU_CUDA"
        #compile pySuperLU          "-j1 --file=gpuMakefile"  "CONFIG+=shellSuperLU_CUDA"

        #compile LA_CUSP            "-j1 --file=cudaMakefile"

        compile LA_Trilinos_Amesos "-j1"
        compile pyTrilinos         "-j1"

        compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellBONMIN"
        compile pyBONMIN           "-j1" "CONFIG+=shellBONMIN"

        compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellIPOPT"
        compile pyBONMIN           "-j1" "CONFIG+=shellIPOPT"

        compile NLOPT_NLPSolver    "-j1"
        compile pyNLOPT            "-j1"
        
        #if [ ${PLATFORM} = "Linux" ]; then
        #  compile LA_Intel_MKL     "-j1"
        #  compile pyIntelPardiso   "-j1"
        #fi
        ;;

  core)  echo [core] ...
         compile dae "-j$Ncpu"
        ;;

  pydae) echo [pydae] ...
         compile pyCore           "-j1"
         compile pyActivity       "-j1"
         compile pyDataReporting  "-j1"
         compile pyIDAS           "-j1"
         compile pyUnits          "-j1"
         ;;

  solvers)  echo [solvers]
            compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU"
            compile pySuperLU          "-j1" "CONFIG+=shellSuperLU"

            compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU_MT"
            compile pySuperLU          "-j1" "CONFIG+=shellSuperLU_MT"

            compile LA_Trilinos_Amesos "-j1"
            compile pyTrilinos         "-j1"
            
            compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellBONMIN"
            compile pyBONMIN           "-j1" "CONFIG+=shellBONMIN"

            compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellIPOPT"
            compile pyBONMIN           "-j1" "CONFIG+=shellIPOPT"

            compile NLOPT_NLPSolver    "-j1"
            compile pyNLOPT            "-j1"

            #if [ ${PLATFORM} = "Linux" ]; then
            #  compile LA_Intel_MKL     "-j1"
            #  compile pyIntelPardiso   "-j1"
            #fi
        ;;

  superlu) echo [superlu]
           compile LA_SuperLU "-j1" "CONFIG+=shellSuperLU"
           compile pySuperLU  "-j1" "CONFIG+=shellSuperLU"
        ;;

  superlu_mt) echo [superlu_mt]
              compile LA_SuperLU "-j1" "CONFIG+=shellSuperLU_MT"
              compile pySuperLU  "-j1" "CONFIG+=shellSuperLU_MT"
        ;;

  superlu_cuda) echo [superlu_cuda]
                compile LA_SUPERLU "-j1 --file=gpuMakefile" "CONFIG+=shellSuperLU_CUDA"
                compile pySuperLU  "-j1 --file=gpuMakefile" "CONFIG+=shellSuperLU_CUDA"
        ;;

  cusp) echo [cusp]
        compile LA_CUSP "-j1 --file=cudaMakefile"
        ;;

  trilinos) echo [trilinos]
            compile LA_Trilinos_Amesos "-j1"
            compile pyTrilinos         "-j1"
        ;;

  intel_pardiso) echo [intel_pardiso]
                      compile LA_Intel_MKL   "-j1"
                      compile pyIntelPardiso "-j1"
        ;;

  bonmin) echo [bonmin]
          compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellBONMIN"
          compile pyBONMIN           "-j1" "CONFIG+=shellBONMIN"
        ;;

  ipopt) echo [ipopt]
         compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellIPOPT"
         compile pyBONMIN           "-j1" "CONFIG+=shellIPOPT"
        ;;

  nlopt) echo [nlopt]
         compile NLOPT_NLPSolver "-j1"
         compile pyNLOPT         "-j1"
        ;;

  *) echo "Unrecognized project: [$*]"
     ;;
esac

cd ${TRUNK}

