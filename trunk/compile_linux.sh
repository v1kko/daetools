#!/bin/sh

set -e

if [ "$1" = "-help" ]; then
  echo "Usage:"
  echo "compile_linux [all]"
  return
fi

PROJECTS=$1
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
  qmake-qt4 $1.pro -r -spec linux-${ARCH} ${CONFIG}
  
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
        #rm -rf *
        cd ${TRUNK}

        #compile dae                "-j$Ncpu"
        compile LA_SuperLU         "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU"
        compile LA_SuperLU         "-j1"                     "CONFIG+=shellCompile CONFIG+=shellSuperLU_MT"
        #compile LA_SUPERLU         "-j1 --file=gpuMakefile"  "CONFIG+=shellCompile CONFIG+=shellSuperLU_CUDA"
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
     for p in $*
     do
       echo Compiling ${p}
       PROJECT=$p
       compile $PROJECT
     done
     ;;
esac

cd ${TRUNK}

