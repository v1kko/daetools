#!/bin/bash

set -e

usage()
{
cat << EOF
usage: $0 [OPTIONS] PROJECT [PROJECT2 PROJECT3 ...]

This script compiles specified daetools libraries and python extension modules.

Typical usage (compiles all daetools libraries, solvers and python extension modules):
    sh compile_linux.sh all
    
Compiling only specified projects:
    sh $0 trilinos superlu nlopt

Achtung, Achtung!!
On MACOS gcc does not work well. llvm-gcc and llvm-g++ should be used.
Add the "llvm-gcc" compiler to the PATH variable if necessary. For instance:
    export PATH=/Developer/usr/bin:$PATH
Make sure there are: QMAKE_CC = llvm-gcc and QMAKE_CXX = llvm-g++ defined in dae.pri
getopt command might be missing - that line should be commented out.

OPTIONS:
   -h | --help                  Show this message.
   
   Python options (if not set use system's default python). One of the following:
    --with-python-binary        Path to the python binary to use.
    --with-python-version       Version of the system's python in the format: major.minor (i.e 2.7).

   Cross compiling options:
    --host          Example: --host i686-w64-mingw32 (defines --host option for cross-compiling with the GNU gcc toolchain).
                    Install the mingw-w64 package from the main Debian repository.
                    CMake uses cross-compile-i686-w64-mingw32.cmake file that targets a toolchain located in /usr/mingw32-i686 directory.
                    and cross-compile-x86_64-w64-mingw32.cmake file that targets a toolchain located in /usr/mingw32-x86_64 directory.
                    Copy trunk/qt-mkspecs/win32-g++-i686-w64-mingw32 to /usr/share/qt/mkspecs
                    
                    Modify dae.pri and set the python major and minor versions.
                    Python root directory must be in the trunk folder: Python[Major][Minor]-[arch] (i.e. Python35-win32).

PROJECT:
    all             Build all daetools c++ libraries, solvers and python extension modules.
                    Equivalent to: dae superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    dae             Build all daetools c++ libraries and python extension modules (no 3rd party LA/(MI)NLP/FE solvers).
                    Equivalent to: units data_reporting idas core activity simulation_loader fmi
    solvers         Build all solvers and their python extension modules.
                    Equivalent to: superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    pydae           Build daetools core python extension modules only.
    
    Individual projects:
        config              Build Config shared c++ library.
        core                Build Core c++ library and its python extension module (pyCore).
        activity            Build Activity c++ library and its python extension module (pyActivity).
        data_reporting      Build DataReporting c++ library and its python extension module (pyDataReporting).
        idas                Build IDAS c++ library and its python extension module (pyIDAS).
        units               Build Units c++ library and its python extension module (pyUnits).
        simulation_loader   Build simulation_loader shared library.
        fmi                 Build FMI wrapper shared library.
        trilinos            Build Trilinos Amesos/AztecOO linear solver and its python extension module (pyTrilinos).
        superlu             Build SuperLU linear solver and its python extension module (pySuperLU).
        superlu_mt          Build SuperLU_MT linear solver and its python extension module (pySuperLU_MT).
        pardiso             Build PARDISO linear solver and its python extension module (pyPardiso).
        intel_pardiso       Build Intel PARDISO linear solver and its python extension module (pyIntelPardiso).
        bonmin              Build BONMIN minlp solver and its python extension module (pyBONMIN).
        ipopt               Build IPOPT nlp solver and its python extension module (pyIPOPT).
        nlopt               Build NLOPT nlp solver and its python extension module (pyNLOPT).
        deal.ii             Build deal.II FEM solvers and its python extension module (pyDealII).
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

  if [ "${DAE_IF_CROSS_COMPILING}" = "1" ]; then
    CONFIG_CROSS_COMPILING="CONFIG+=crossCompile"
  fi
  
  echo ""
  echo "[*] Configuring the project with ($2, $3)..."
  echo "${QMAKE} -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile ${CONFIG_CROSS_COMPILING} customPython="${PYTHON}" -spec ${QMAKE_SPEC} ${CONFIG}"
  
  ${QMAKE} -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile ${CONFIG_CROSS_COMPILING} customPython="${PYTHON}" -spec ${QMAKE_SPEC} ${CONFIG}

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
QMAKE="qmake-qt4"
QMAKE_SPEC="linux-g++"
DAE_IF_CROSS_COMPILING=0

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  # If there are problems with memory and speed of compilation set:
  # Ncpu=1
  QMAKE="qmake"
  QMAKE_SPEC=macx-g++
elif [ ${PLATFORM} = "Linux" ]; then
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
  QMAKE_SPEC=linux-g++
else
  Ncpu=4
  QMAKE_SPEC=win32-g++
fi

if [ ${Ncpu} -gt 1 ]; then
  Ncpu=$(($Ncpu+1))
fi

cd "${TRUNK}"

if [ ! -d release ]; then
    mkdir release
fi

if [ ${PLATFORM} = "Darwin" ]; then
  args= 
else
  args=`getopt -a -o "h" -l "help,with-python-binary:,with-python-version:,host:" -n "compile_linux" -- $*`
fi

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

       --host) DAE_IF_CROSS_COMPILING=1
               QMAKE="qmake-qt4"
               if [ "$2" = "i686-w64-mingw32" ]; then
                  QMAKE_SPEC="win32-g++-i686-w64-mingw32"
               elif [ "$2" = "x86_64-w64-mingw32" ]; then
                  QMAKE_SPEC="win64-g++-x86_64-w64-mingw32"
               else
                  QMAKE_SPEC=
               fi
               PYTHON=""
               shift ; shift
               ;;
                                    
    --) shift; break 
        ;;
  esac
done

# Check if any project is specified
if [ -z "$1" ]; then
  usage
  exit
fi

# Check if requested projects exist
for project in "$@"
do
  case "$project" in
    all)              ;;
    config)           ;;
    core)             ;;
    activity)         ;;
    data_reporting)   ;;
    idas)             ;;
    units)            ;;
    simulation_loader);;
    fmi)              ;;
    dae)              ;;
    pydae)            ;;
    solvers)          ;;
    trilinos)         ;;
    superlu)          ;;
    superlu_mt)       ;;
    pardiso)          ;;
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
echo "  - Qmake-spec:           ${QMAKE_SPEC}"
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
                #  compile LA_Pardiso  "-j1"
                #  compile pyPardiso   "-j1"
                #fi

                #if [ ${PLATFORM} = "Linux" ]; then
                #  compile LA_Intel_MKL     "-j1"
                #  compile pyIntelPardiso   "-j1"
                #fi
                ;;
        
        config) compile Config "-j1"
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

        simulation_loader)  compile simulation_loader   "-j1"
                            ;;
                            
        fmi)  compile fmi "-j1"
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
                    #  compile LA_Pardiso  "-j1"
                    #  compile pyPardiso   "-j1"
                    #fi

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

        pardiso) compile LA_Pardiso "-j1"
                 compile pyPardiso  "-j1"
                 ;;

        intel_pardiso) compile LA_IntelPardiso "-j1"
                       compile pyIntelPardiso  "-j1"
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

