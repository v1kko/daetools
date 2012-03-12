#!/bin/sh

set -e

if [ "$1" = "-help" ]; then
  echo "Usage:"
  echo "compile_linux all | dae | trilinos | superlu | superlu_mt | superlu_cuda | bonmin | ipopt | nlopt | cusp"
  return
fi

if [ "$1" = "" ]; then
  PROJECTS=all
else
  PROJECTS=$1
fi

HOST_ARCH=`uname -m`
PLATFORM=`uname -s`
TRUNK="$( cd "$( dirname "$0" )" && pwd )"

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  SPEC=macx-g++
else
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
  Ncpu=$(($Ncpu+1))
  if [ ${HOST_ARCH} = "x86_64" ]; then
    SPEC=linux-g++-64
  else
    SPEC=linux-g++-32
  fi
fi

cd ${TRUNK}

compile () {
  DIR=$1
  MAKEARG=$2
  CONFIG=$3
  echo "Compiling the project $DIR ..."

  if [ ${DIR} = "dae" ]; then
    cd ${TRUNK}
  else
    cd ${DIR}
  fi

  echo 
  echo "*** EXECUTE: qmake $1.pro -r CONFIG+=release -spec ${SPEC} ${CONFIG}"
  echo 
  qmake $1.pro -r CONFIG+=release -spec ${SPEC} ${CONFIG}
  
  echo 
  echo "*** EXECUTE: make clean -w"
  echo 
  make clean -w
  
  echo 
  echo "*** EXECUTE: make ${MAKEARG} -w"
  echo 
  make ${MAKEARG} -w
  
  cd ${TRUNK}
  echo 
}

case ${PROJECTS} in
  all)  echo Compile ALL projects
        if [ ! -d ${TRUNK}/release ]; then
          mkdir ${TRUNK}/release
        fi
        cd ${TRUNK}/release
        rm -rf *
        cd ${TRUNK}

        compile dae                "-j$Ncpu"

        compile LA_SuperLU         "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU"
        compile pySuperLU          "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU"

        compile LA_SuperLU         "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU_MT"
        compile pySuperLU          "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU_MT"

        #compile LA_SUPERLU         "-j1 --file=gpuMakefile"  "CONFIG+=shellCompile CONFIG+=shellSuperLU_CUDA"
        #compile pySuperLU          "-j1 --file=gpuMakefile"  "CONFIG+=shellCompile CONFIG+=shellSuperLU_CUDA"

        #compile LA_CUSP            "-j1 --file=cudaMakefile"

        compile LA_Trilinos_Amesos "-j1"

        compile BONMIN_MINLPSolver "-j1"                     "CONFIG+=shellCompile CONFIG+=shellBONMIN"
        compile pyBONMIN           "-j1"                     "CONFIG+=shellCompile CONFIG+=shellBONMIN"

        compile BONMIN_MINLPSolver "-j1"                     "CONFIG+=shellCompile CONFIG+=shellIPOPT"
        compile pyBONMIN           "-j1"                     "CONFIG+=shellCompile CONFIG+=shellIPOPT"

        compile NLOPT_NLPSolver    "-j1"
        compile pyNLOPT            "-j1"
        ;;

  *) echo Compile projects: [$*] 
     for PROJECT in $*
     do
       echo Compiling $PROJECT...
       if [ ${PROJECT} = "dae" ]; then
         compile dae "-j$Ncpu"
       
       elif [ ${PROJECT} = "superlu" ]; then
         compile LA_SuperLU "-j1" "CONFIG+=shellCompile CONFIG+=shellSuperLU"
         compile pySuperLU  "-j1" "CONFIG+=shellCompile CONFIG+=shellSuperLU"

       elif [ ${PROJECT} = "superlu_mt" ]; then
         compile LA_SuperLU "-j1" "CONFIG+=shellCompile CONFIG+=shellSuperLU_MT"
         compile pySuperLU  "-j1" "CONFIG+=shellCompile CONFIG+=shellSuperLU_MT"

       elif [ ${PROJECT} = "superlu_cuda" ]; then
         compile LA_SUPERLU "-j1 --file=gpuMakefile" "CONFIG+=shellCompile CONFIG+=shellSuperLU_CUDA"
         compile pySuperLU  "-j1 --file=gpuMakefile" "CONFIG+=shellCompile CONFIG+=shellSuperLU_CUDA"

       elif [ ${PROJECT} = "cusp" ]; then
         compile LA_CUSP "-j1 --file=cudaMakefile"

       elif [ ${PROJECT} = "trilinos" ]; then
         compile LA_Trilinos_Amesos "-j1"

       elif [ ${PROJECT} = "bonmin" ]; then
         compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellCompile CONFIG+=shellBONMIN"
         compile pyBONMIN           "-j1" "CONFIG+=shellCompile CONFIG+=shellBONMIN"

       elif [ ${PROJECT} = "ipopt" ]; then
         compile BONMIN_MINLPSolver "-j1" "CONFIG+=shellCompile CONFIG+=shellIPOPT"
         compile pyBONMIN           "-j1" "CONFIG+=shellCompile CONFIG+=shellIPOPT"

       elif [ ${PROJECT} = "nlopt" ]; then
         compile NLOPT_NLPSolver "-j1"
         compile pyNLOPT         "-j1"

       else
         echo "Unrecognized project: ${PROJECT}"
       fi
     done
     ;;
esac

cd ${TRUNK}

