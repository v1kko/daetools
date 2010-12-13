############################################################################
#
#  Program:         SuperLU_MT
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   August 15, 1997
#
#  Modified:	    November 25, 2010 version 1.0
#
############################################################################
#
#  The machine (platform) identifier to append to the library names
#
PLAT = 

#
#  The name of the libraries to be created/linked to
#
TMGLIB       	= libtmglib$(PLAT).a
SUPERLULIB   	= libsuperlu_mt$(PLAT).a
BLASDEF	        = -DUSE_VENDOR_BLAS
BLASLIB		= -lblas
MATHLIB		= -lm 
MPLIB		= -lgomp -lpthread

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH          	= ar
ARCHFLAGS     	= cr
RANLIB       	= ranlib
#
# Definitions used by CPP:
PREDEFS      	= -D__OPENMP
#
CC           	= gcc
CFLAGS       	= $(PREDEFS) -O0
NOOPTS	     	=
BLASOPTS	=
LOADER       	= gcc
LOADOPTS     	=
FORTRAN		= g77
FFLAGS		= -O -fopenmp

#
#  C preprocessor defs for compilation (-DNoChange, -DAdd_, or -DUpCase)
#
CDEFS        	= -DAdd_
