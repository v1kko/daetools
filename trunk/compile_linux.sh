#!/bin/bash

set -e

usage()
{
cat << EOF
usage: $0 [options] projects

This script compiles specified daetools libraries and python extension modules.

Typical use (compiles all daetools libraries, solvers and python extension modules):
    sh compile_linux.sh all

OPTIONS:
   -h | --help                      Show this message.
   One of the following:
        --with-python-binary        Path to python binary to use.
        --with-python-version       Version of the system's python.
                                    Format: major.minor (i.e 2.7).

PROJECTS:
    all             Build all daetools c++ libraries, solvers and python extension modules.
                    Equivalent to: dae superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    dae             Build all daetools c++ libraries and python extension modules (no 3rd party LA/(MI)NLP/FE solvers).
                    Equivalent to: units data_reporting idas core activity
    solvers         Build all solvers and their python extension modules.
                    Equivalent to: superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    pydae           Build daetools core python extension modules only.
    
    Individual projects:
    core            Build Core c++ library and its python extension module.
    activity        Build Activity c++ library and its python extension module.
    data_reporting  Build DataReporting c++ library and its python extension module.
    idas            Build IDAS c++ library and its python extension module.
    units           Build Units c++ library and its python extension module.
    trilinos        Build Trilinos Amesos/AztecOO linear solver and its python extension module.
    superlu         Build SuperLU linear solver and its python extension module.
    superlu_mt      Build SuperLU_MT linear solver and its python extension module.
    intel_pardiso   Build Intel PARDISO linear solver and its python extension module.
    bonmin          Build BONMIN minlp solver and its python extension module.
    ipopt           Build IPOPT nlp solver and its python extension module.
    nlopt           Build NLOPT nlp solver and its python extension module.
    deal.ii         Build deal.II FEM solvers and its python extension module.
EOF
}

compile() 
{
  DIR=$1
  MAKEARG=$2
  CONFIG=$3
  echo ""
  echo "*******************************************************************************"
  echo "                Build the project: $DIR"
  echo "*******************************************************************************"
  
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

  qmake-qt4 -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile "customPython=${PYTHON}" -spec ${SPEC} ${CONFIG}
  
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
  cd "${TRUNK}"
}

# Default python binary:
PYTHON=`python -c "import sys; print(sys.executable)"`
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

cd "${TRUNK}"

if [ ! -d release ]; then
    mkdir release
fi

args=`getopt -a -o "h" -l "help,with-python-binary:,with-python-version:" -n "compile_linux" -- $*`

# Process options
for i; do
  case "$i" in
    -h|--help)  usage
                exit 1
                ;;
                  
       --with-python-binary)  PYTHON=`$2 -c "import sys; print(sys.executable)"`
                              shift ; shift 
                              ;;
                            
       --with-python-version )  PYTHON=`python$2 -c "import sys; print(sys.executable)"`
                                shift ; shift 
                                ;;
                                    
    --) shift; break 
        ;;
  esac
done

# Check if any project is specified
if [ -z "$@" ]; then
  usage
  exit
fi

# Check if requested projects exist
for project in "$@"
do
  case "$project" in
    all)              ;;
    core)             ;;
    activity)         ;;
    data_reporting)   ;;
    idas)             ;;
    units)            ;;
    dae)              ;;
    pydae)            ;;
    solvers)          ;;
    trilinos)         ;;
    superlu)          ;;
    superlu_mt)       ;;
    intel_pardiso)    ;;
    bonmin)           ;;
    ipopt)            ;;
    nlopt)            ;; 
    deal.ii)          ;; 
    *) echo Unrecognized project: "$project"
       exit
       ;;
  esac
done

echo ""
echo "###############################################################################"
echo "Proceed with the following options:"
echo "  - Qmake-spec:           ${SPEC}"
echo "  - Python:               ${PYTHON}"
echo "  - Platform:             $PLATFORM"
echo "  - Architecture:         $HOST_ARCH"
echo "  - Number of threads:    ${Ncpu}"
echo "  - Projects to compile:  $@"
echo "###############################################################################"
echo ""

# Process arguments
for project in "$@"
do
    case "$project" in
        all)    cd ${TRUNK}/release
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
                
                compile pyDealII  "-j1"
                
                #if [ ${PLATFORM} = "Linux" ]; then
                #  compile LA_Intel_MKL     "-j1"
                #  compile pyIntelPardiso   "-j1"
                #fi
                ;;
        
        core)  compile Core   "-j5"
               compile pyCore "-j1"
               ;;
        
        activity) compile Activity   "-j5"
                  compile pyActivity "-j1"
                  ;;
                  
        data_reporting) compile DataReporting   "-j5"
                        compile pyDataReporting "-j1"
                        ;;
                        
        idas)   compile IDAS_DAESolver "-j5"
                compile pyIDAS         "-j1"
                ;;
        
        units)  compile Units   "-j5"
                compile pyUnits "-j1"
                ;;

        dae)    compile dae "-j$Ncpu"
                ;;

        pydae)  compile pyCore           "-j1"
                compile pyActivity       "-j1"
                compile pyDataReporting  "-j1"
                compile pyIDAS           "-j1"
                compile pyUnits          "-j1"
                ;;

        solvers)    compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU"
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

                    compile pyDealII           "-j1"
                     
                    #if [ ${PLATFORM} = "Linux" ]; then
                    #  compile LA_Intel_MKL     "-j1"
                    #  compile pyIntelPardiso   "-j1"
                    #fi
                    ;;

        superlu)    compile LA_SuperLU "-j1" "CONFIG+=shellSuperLU"
                    compile pySuperLU  "-j1" "CONFIG+=shellSuperLU"
                    ;;

        superlu_mt) compile LA_SuperLU "-j1" "CONFIG+=shellSuperLU_MT"
                    compile pySuperLU  "-j1" "CONFIG+=shellSuperLU_MT"
                    ;;
  
        superlu_cuda) compile LA_SUPERLU "-j1 --file=gpuMakefile" "CONFIG+=shellSuperLU_CUDA"
                      compile pySuperLU  "-j1 --file=gpuMakefile" "CONFIG+=shellSuperLU_CUDA"
                      ;;

        cusp) compile LA_CUSP "-j1 --file=cudaMakefile"
              ;;

        trilinos) compile LA_Trilinos_Amesos "-j1"
                  compile pyTrilinos         "-j1"
                  ;;

        intel_pardiso) compile LA_Intel_MKL   "-j1"
                       compile pyIntelPardiso "-j1"
                       ;;

        bonmin) compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellBONMIN"
                compile pyBONMIN           "-j1" "CONFIG+=shellBONMIN"
                ;;

        ipopt) compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellIPOPT"
               compile pyBONMIN           "-j1" "CONFIG+=shellIPOPT"
               ;;

        nlopt) compile NLOPT_NLPSolver "-j1"
               compile pyNLOPT         "-j1"
               ;;

        deal.ii) compile pyDealII  "-j1"
                 ;;
               
        *) echo "??????????????????????"
           echo Unrecognized project: "$project"
           echo "??????????????????????"
           ;;
    esac
done

cd ${TRUNK}

