# -----------------------------------------------------------------
# $Revision: 1.8 $
# $Date: 2007/01/29 17:36:28 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# -----------------------------------------------------------------
# Makefile for parallel NVECTOR module
#
# src/nvec_par/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/bash



srcdir       = .
builddir     = .
abs_builddir = /home/ciroki/Data/dae/trunk/ida-2.6.0/src/nvec_par
top_builddir = ../..
prefix       = /usr/local
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

INSTALL        = /usr/bin/install -c
INSTALL_LIB    = ${INSTALL}
INSTALL_HEADER = ${INSTALL} -m 644

LIBTOOL      = $(SHELL) $(top_builddir)/libtool
LIBTOOL_DEPS = config/ltmain.sh

MPICC       = /usr/bin/mpicc
MPI_INC_DIR = /usr/bin/../include
MPI_LIB_DIR = /usr/bin/../lib
MPI_LIBS    = 
MPI_FLAGS   = 
CPPFLAGS    = 
CFLAGS      = -g -O2
LDFLAGS     = 
LIBS        = -lm 

FCMIX_ENABLED = no

top_srcdir = $(srcdir)/../..

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(MPI_INC_DIR)

LIB_REVISION = 0:2:0

NVECPAR_LIB       = libsundials_nvecparallel.la
NVECPAR_LIB_FILES = nvector_parallel.lo

FNVECPAR_LIB       = libsundials_fnvecparallel.la
FNVECPAR_LIB_FILES = fnvector_parallel.lo

SHARED_LIB_FILES = $(top_builddir)/src/sundials/sundials_math.lo

mkinstalldirs = $(SHELL) $(top_srcdir)/config/mkinstalldirs
rminstalldirs = $(SHELL) $(top_srcdir)/config/rminstalldirs

all: $(NVECPAR_LIB) $(FNVECPAR_LIB)

$(NVECPAR_LIB): shared $(NVECPAR_LIB_FILES)
	$(LIBTOOL) --mode=link $(MPICC) $(CFLAGS) $(MPI_FLAGS) -o $(NVECPAR_LIB) $(NVECPAR_LIB_FILES) $(SHARED_LIB_FILES) $(LDFLAGS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS) -rpath $(libdir) -version-info $(LIB_REVISION)

$(FNVECPAR_LIB): $(FNVECPAR_LIB_FILES)
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                                                                                                                                                                                    \
	   echo "${LIBTOOL} --mode=link ${MPICC} ${CFLAGS} ${MPI_FLAGS} -o ${FNVECPAR_LIB} ${FNVECPAR_LIB_FILES} ${SHARED_LIB_FILES} ${LDFLAGS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} -rpath ${libdir} -version-info ${LIB_REVISION}" ; \
	   ${LIBTOOL} --mode=link ${MPICC} ${CFLAGS} ${MPI_FLAGS} -o ${FNVECPAR_LIB} ${FNVECPAR_LIB_FILES} ${SHARED_LIB_FILES} ${LDFLAGS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} -rpath ${libdir} -version-info ${LIB_REVISION} ;        \
	 fi

install: $(NVECPAR_LIB) $(FNVECPAR_LIB)
	$(mkinstalldirs) $(includedir)/nvector
	$(mkinstalldirs) $(libdir)
	$(LIBTOOL) --mode=install $(INSTALL_LIB) $(NVECPAR_LIB) $(libdir)
	$(INSTALL_HEADER) $(top_srcdir)/include/nvector/nvector_parallel.h $(includedir)/nvector/
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                                    \
	   echo "${LIBTOOL} --mode=install ${INSTALL_LIB} ${FNVECPAR_LIB} ${libdir}" ; \
	   ${LIBTOOL} --mode=install ${INSTALL_LIB} ${FNVECPAR_LIB} ${libdir} ;        \
	 fi

uninstall:
	$(LIBTOOL) --mode=uninstall rm -f $(libdir)/$(NVECPAR_LIB)
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                             \
	   echo "${LIBTOOL} --mode=uninstall rm -f ${libdir}/${FNVECPAR_LIB}" ; \
	   ${LIBTOOL} --mode=uninstall rm -f ${libdir}/${FNVECPAR_LIB} ;        \
	 fi
	rm -f $(includedir)/nvector/nvector_parallel.h
	$(rminstalldirs) ${includedir}/nvector

shared:
	@cd ${top_builddir}/src/sundials ; \
	 ${MAKE} ;                         \
	 cd ${abs_builddir}

clean:
	$(LIBTOOL) --mode=clean rm -f $(NVECPAR_LIB)
	rm -f $(NVECPAR_LIB_FILES)
	rm -f nvector_parallel.o
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then               \
	   echo "${LIBTOOL} --mode=clean rm -f ${FNVECPAR_LIB}" ; \
	   ${LIBTOOL} --mode=clean rm -f ${FNVECPAR_LIB} ;        \
	   echo "rm -f ${FNVECPAR_LIB_FILES}" ;                   \
	   rm -f ${FNVECPAR_LIB_FILES} ;                          \
	   echo "rm -f fnvector_parallel.o" ;                     \
	   rm -f fnvector_parallel.o ;                            \
	 fi

distclean: clean
	rm -f Makefile

nvector_parallel.lo: $(srcdir)/nvector_parallel.c
	$(LIBTOOL) --mode=compile $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/nvector_parallel.c
fnvector_parallel.lo: $(srcdir)/fnvector_parallel.c
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                                                                                    \
	   echo "${LIBTOOL} --mode=compile ${MPICC} ${CPPFLAGS} ${MPI_FLAGS} $(INCLUDES) ${CFLAGS} -c ${srcdir}/fnvector_parallel.c" ; \
	   ${LIBTOOL} --mode=compile ${MPICC} ${CPPFLAGS} ${MPI_FLAGS} $(INCLUDES) ${CFLAGS} -c ${srcdir}/fnvector_parallel.c ;        \
	 fi

libtool: $(top_builddir)/$(LIBTOOL_DEPS)
	@cd ${top_builddir} ;                 \
	 ${SHELL} ./config.status --recheck ; \
	 cd ${abs_builddir}
