#!/bin/sh

set -e

if [ "$1" = "-help" ]; then
  echo "Usage:"
  echo "compile_linux [all]"
  return
fi

if [ $1 = "" ]; then
  PROJECTS=all
else
  PROJECTS=$1
fi
HOST_ARCH=`uname -m`
TRUNK=`pwd`
Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
Ncpu=$(($Ncpu+1))

compile () {
  DIR=$1
  MAKEARG=$2
  CONFIG=$3
  echo "Compiling the project $DIR ..."

  if [ ${HOST_ARCH} = "x86_64" ]; then
    ARCH=g++-64
  else
    ARCH=g++
  fi

  if [ ${DIR} = "dae" ]; then
    cd ${TRUNK}
  else
    cd ${DIR}
  fi

  echo 
  echo "*** EXECUTE: qmake-qt4 $1.pro -r -spec linux-${ARCH} ${CONFIG}"
  echo 
  qmake-qt4 $1.pro -r CONFIG+=release -spec linux-${ARCH} ${CONFIG}
  
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

