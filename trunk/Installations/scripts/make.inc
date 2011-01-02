############################################################################
#
#  Program:         SuperLU
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   December 29, 2010
#
#  Modified:	    December 25, 2010
#
############################################################################
#
#  The machine (platform) identifier to append to the library names
#
PLAT = 

#
#  The name of the libraries to be created/linked to
#
MAJOR        = 4
MINOR        = 1
VERSION      = $(MAJOR).$(MINOR)

SuperLUroot	    = $(HOME)/Data/daetools/trunk/superlu
SUPERLULIB      = $(SuperLUroot)/lib/libsuperlu_$(VERSION).a
SUPERLULIB_SH   = $(SuperLUroot)/lib/libsuperlu.so.$(VERSION)
SUPERLULIB_LN   = $(SuperLUroot)/lib/libsuperlu.so.$(MAJOR)
CDAE_SUPERLULIB = $(SuperLUroot)/lib/libcdaesuperlu.so

## ATLAS BLAS causes single-precision to fail
BLASDEF = -DUSE_VENDOR_BLAS
BLASLIB = -lblas

TMGLIB     	= libtmglib.a
LIBS		= $(SUPERLULIB) $(SUPERLULIB_SH) $(SUPERLULIB_LN) $(BLASLIB)

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = ar
ARCHFLAGS    = cr
RANLIB       = ranlib

LDFLAGS   = -s
CC        = gcc
CFLAGS    = -DPRNTlevel=0 -O3 -fPIC -pipe
NOOPTS    = -fPIC

FORTRAN	     = gfortran
FFLAGS       = -O2
LOADER       = $(FORTRAN)
LOADOPTS     =

#
#  C preprocessor defs for compilation for the Fortran interface
#  (-DNoChange, -DAdd_, -DAdd__, or -DUpCase)
#
CDEFS        = -DAdd_
#
# The directory in which Matlab is installed
#
MATLAB	     = 

