#!/bin/bash

set -e

usage()
{
cat << EOF
usage: $0 [OPTIONS] PROJECT [PROJECT2 PROJECT3 ...]

This script compiles specified daetools libraries and python extension modules.

Typical usage (compiles all daetools libraries, solvers and python extension modules):
    sh compile.sh all
    
Compiling only specified projects:
    sh compile.sh trilinos superlu nlopt

Achtung, Achtung!!
On MACOS gcc should be used (the XCode does not provide Fortran compiler).
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
                    Copy trunk/qt-mkspecs/win32-g++-i686-w64-mingw32 to /usr/share/qt/mkspecs (or i.e. /usr/lib/x86_64-linux-gnu/qt5/mkspecs)
                    
                    Modify dae.pri and set the python major and minor versions.
                    Python root directory must be in the trunk folder: Python[Major][Minor]-[arch] (i.e. Python35-win32).

PROJECT:
    all             Build all daetools c++ libraries, solvers and python extension modules.
                    On GNU/Linux and macOS equivalent to: dae superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
                    On Windows equivalent to: dae superlu trilinos ipopt bonmin nlopt deal.ii
    dae             Build all daetools c++ libraries and python extension modules (no 3rd party LA/(MI)NLP/FE solvers).
                    Equivalent to: config cool_prop units data_reporting idas core activity simulation_loader fmi
    solvers         Build all solvers and their python extension modules.
                    On GNU/Linux and macOS equivalent to: superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
                    On Windows equivalent to: superlu trilinos ipopt bonmin nlopt deal.ii
    pydae           Build daetools core python extension modules only.
    
    Individual projects:
        config                  Build Config shared c++ library.
        core                    Build Core c++ library and its python extension module (pyCore).
        activity                Build Activity c++ library and its python extension module (pyActivity).
        data_reporting          Build DataReporting c++ library and its python extension module (pyDataReporting).
        idas                    Build IDAS c++ library and its python extension module (pyIDAS).
        units                   Build Units c++ library and its python extension module (pyUnits).
        simulation_loader       Build simulation_loader shared library.
        fmi                     Build FMI wrapper shared library.
        fmi_ws                  Build FMI wrapper shared library that uses daetools FMI web service.
        trilinos                Build Trilinos Amesos/AztecOO linear solver and its python extension module (pyTrilinos).
        superlu                 Build SuperLU linear solver and its python extension module (pySuperLU).
        superlu_mt              Build SuperLU_MT linear solver and its python extension module (pySuperLU_MT).
        pardiso                 Build PARDISO linear solver and its python extension module (pyPardiso).
        intel_pardiso           Build Intel PARDISO linear solver and its python extension module (pyIntelPardiso).
        bonmin                  Build BONMIN minlp solver and its python extension module (pyBONMIN).
        ipopt                   Build IPOPT nlp solver and its python extension module (pyIPOPT).
        nlopt                   Build NLOPT nlp solver and its python extension module (pyNLOPT).
        deal.ii                 Build deal.II FEM library and its python extension module (pyDealII).
        cool_prop               Build CoolProp thermo package (cdaeCoolPropThermoPackage).
        cape_open_thermo        Build Cape Open thermo-physical property package library (cdaeCapeOpenThermoPackage.dll, Windows only).
        opencl_evaluator        Build Evaluator_OpenCL library and its python extension module (pyEvaluator_OpenCL).
        pyopencs                Build pyOpenCS python extension module (pyOpenCS).
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
  
  if [ ${PLATFORM} = "Windows" ]; then
    MAKEARG=
  fi

  echo ""
  echo "[*] Configuring the project with ($2, $3)..."
  echo "${QMAKE} -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile ${CONFIG_CROSS_COMPILING} customPython="${PYTHON}" -spec ${QMAKE_SPEC} ${CONFIG}"
  
  ${QMAKE} -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile ${CONFIG_CROSS_COMPILING} customPython="${PYTHON}" -spec ${QMAKE_SPEC} ${CONFIG}

  echo ""
  echo "[*] Cleaning the project..."
  echo ""
  if [ ${PLATFORM} = "Windows" ]; then
    ${MAKE} clean
  else
    ${MAKE} clean -w
  fi
  echo ""
  echo "[*] Compiling the project..."
  echo ""
  if [ ${PLATFORM} = "Windows" ]; then
    ${MAKE} ${MAKEARG}
  else
    ${MAKE} ${MAKEARG} -w
  fi

  echo ""
  echo "[*] Installing the project..."
  echo ""
  # qmake INSTALLS ignore rules if .files are not existing.
  # Therefore, run qmake again after build (when all files exist) to create a makefile with all rules included.
  ${QMAKE} -makefile $1.pro -r CONFIG+=release CONFIG+=silent CONFIG+=shellCompile ${CONFIG_CROSS_COMPILING} customPython="${PYTHON}" -spec ${QMAKE_SPEC} ${CONFIG}
  ${MAKE} install
  
  echo ""
  echo "[*] Done!"
  cd "${TRUNK}"
}

compile_cape_open_thermo()
{
  if [[ "${PLATFORM}" == "Windows" ]]; then
    echo ""
    echo "[*] Compiling cape_open_thermo..."
    echo ""
    
    cd CapeOpenThermoPackage
    
    if [[ "${HOST_ARCH}" == "win32" ]]; then
      MS_BUILD_PLATFORM="x86"
    elif [[ "${HOST_ARCH}" == "win64" ]]; then
      MS_BUILD_PLATFORM="x64"
    else
      echo unknown HOST_ARCH: $HOST_ARCH
      exit 1
    fi
    
    echo "msbuild.exe CapeOpenThermoPackage.vcxproj /target:rebuild /p:Platform=${MS_BUILD_PLATFORM} /p:Configuration=Release /p:PlatformToolset=v140 /p:UseEnv=true"
    msbuild.exe CapeOpenThermoPackage.vcxproj -target:rebuild -p:Platform="${MS_BUILD_PLATFORM}" -p:Configuration="Release" -p:PlatformToolset="v140" -p:UseEnv=true

    cp -fa ../release/cdaeCapeOpenThermoPackage.lib ${DAE_DEV_LIB_DIR}/cdaeCapeOpenThermoPackage.lib
    cp -fa ../release/cdaeCapeOpenThermoPackage.dll ${DAE_DEV_LIB_DIR}/cdaeCapeOpenThermoPackage.dll
    cp -fa ../release/cdaeCapeOpenThermoPackage.dll ${SOLIBS_DIR}/cdaeCapeOpenThermoPackage.dll
  fi
  cd "${TRUNK}"
}

# Default python binary:
PYTHON=`python -c "import sys; print(sys.executable)"`
HOST_ARCH=`uname -m`
PLATFORM=`uname -s`
TRUNK="$( cd "$( dirname "$0" )" && pwd )"
QMAKE="qmake"
QMAKE_SPEC="linux-g++"
DAE_IF_CROSS_COMPILING=0

if [[ ${PLATFORM} == *"MSYS_"* ]]; then
  PLATFORM="Windows"
  # Platform should be set by i.e. vcbuildtools.bat 
  VC_PLAT=`cmd "/C echo %Platform% "`
  echo $VC_PLAT
  if [[ ${VC_PLAT} == *"X86"* ]]; then
    HOST_ARCH="win32"
  elif [[ ${VC_PLAT} == *"x86"* ]]; then
    HOST_ARCH="win32"
  elif [[ ${VC_PLAT} == *"x64"* ]]; then
    HOST_ARCH="win64"
  elif [[ "${VC_PLAT}" == *"X64"* ]]; then
    HOST_ARCH="win64"
  else
    echo unknown HOST_ARCH: $HOST_ARCH
    exit 1
  fi
fi

SOLIBS_DIR="${TRUNK}/daetools-package/daetools/solibs/${PLATFORM}_${HOST_ARCH}/lib"
if [ ! -e ${SOLIBS_DIR} ]; then
    mkdir -p ${SOLIBS_DIR}
fi
DAE_DEV_LIB_DIR="${TRUNK}/daetools-dev/${PLATFORM}_${HOST_ARCH}/lib"
if [ ! -e ${DAE_DEV_LIB_DIR} ]; then
    mkdir -p ${DAE_DEV_LIB_DIR}
fi

if [ ${PLATFORM} = "Darwin" ]; then
  #Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  # If there are problems with memory and speed of compilation set:
  Ncpu=2
  QMAKE="qmake"
  QMAKE_SPEC=macx-g++
  export CC=/usr/local/bin/gcc-8
  export CXX=/usr/local/bin/g++-8
  export CPP=/usr/local/bin/cpp-8
  export LD=/usr/local/bin/gcc-8
  export F77=/usr/local/bin/gfortran-8

  alias gcc=/usr/local/bin/gcc-8
  alias g++=/usr/local/bin/g++-8
  alias cc=/usr/local/bin/gcc-8
  alias c++=/usr/local/bin/c++-8
  alias ld=/usr/local/bin/gcc-8
  
elif [ ${PLATFORM} = "Linux" ]; then
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
  QMAKE="/usr/bin/qmake"
  QMAKE_SPEC=linux-g++
  
elif [ ${PLATFORM} = "Windows" ]; then
  Ncpu=4
  QMAKE_SPEC=win32-msvc
  
else
  Ncpu=4
  QMAKE_SPEC=win32-g++
fi

if [ ${Ncpu} -gt 1 ]; then
  Ncpu=$(($Ncpu+1))
fi

MAKE="make"
if [ ${PLATFORM} = "Windows" ]; then
  MAKE="nmake"
fi

cd "${TRUNK}"

if [ ! -d release ]; then
    mkdir release
fi

if [ ${PLATFORM} = "Darwin" ]; then
  args= 
else
  args=`getopt -a -o "h" -l "help,with-python-binary:,with-python-version:,host:" -n "compile" -- $*`
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
            QMAKE="qmake"
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
    fmi_ws)           ;;
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
    cool_prop)        ;;
    cape_open_thermo) ;; 
    opencl_evaluator) ;;
    pyopencs)           ;;
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

                compile_cape_open_thermo
                
                compile dae                "-j$Ncpu" 

                compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU"
                compile pySuperLU          "-j1" "CONFIG+=shellSuperLU"

                if [ ${PLATFORM} != "Windows" ]; then
                  compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU_MT"
                  compile pySuperLU          "-j1" "CONFIG+=shellSuperLU_MT"
                fi

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
                            
        fmi_ws)  compile fmi_ws "-j1"
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

                    if [ ${PLATFORM} != "Windows" ]; then
                      compile LA_SuperLU         "-j1" "CONFIG+=shellSuperLU_MT"
                      compile pySuperLU          "-j1" "CONFIG+=shellSuperLU_MT"
                    fi

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
        
        cool_prop) compile CoolPropThermoPackage "-j1"
                ;;
            
        cape_open_thermo) compile_cape_open_thermo
                          ;;
        
        opencl_evaluator) compile pyEvaluator_OpenCL "-j1"
                          ;;
                          
        pyopencs) compile pyOpenCS "-j1"
                  ;;
                          
        *) echo "??????????????????????"
           echo Unrecognized project: "$project"
           echo "??????????????????????"
           ;;
    esac
done

cd ${TRUNK}

