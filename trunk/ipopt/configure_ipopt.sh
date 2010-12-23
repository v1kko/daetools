#!/bin/sh

# Example call:
# ./configure_ipopt.sh /home/ciroki/Data/daetools/trunk

if [ "$1" == "" ]; then
  echo "You have to specify daetools/trunk folder! For example:"
  echo "./configure_ipopt.sh ~/Data/daetools/trunk"
  exit
fi

ROOT="$1"
IPOPT="$ROOT/ipopt"
MUMPS="$ROOT/mumps"
BUILD="$IPOPT/build"

if [ ! -d $BUILD ]; then
  mkdir $BUILD
fi
cd $BUILD

$IPOPT/configure --with-pic --enable-static=yes --enable-shared=no --with-mumps-lib="$MUMPS/lib/libdmumps.a $MUMPS/lib/libmumps_common.a $MUMPS/libseq/libmpiseq.a $MUMPS/lib/libpord.a " --with-mumps-incdir=$MUMPS/include

cd $BUILD
