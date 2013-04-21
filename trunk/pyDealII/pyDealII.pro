include(../dae.pri)
QT -= core gui
TARGET = pyDealII
TEMPLATE = lib

DEALII_DIR     = ../deal.II
TBB_DIR        = ../deal.II/contrib/tbb/tbb30_104oss
DEALII_INCLUDE = $${DEALII_DIR}/include \
                 $${TBB_DIR}/include
DEALII_LIB_DIR = $${DEALII_DIR}/lib
DEALII_LIBS    = -ldeal_II -ltbb -lz -lblas -lgfortran

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
               $${DEALII_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${DEALII_LIB_DIR}

QMAKE_CXXFLAGS += -std=c++0x

#QMAKE_CXXFLAGS += -DHAVE_CONFIG_H -DHAVE_ISNAN -ggdb  -DBOOST_NO_HASH -DBOOST_NO_SLIST -DDEBUG -Wall -W -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-long-long -std=c++0x -Wa,--compress-debug-sections -pthread -D_REENTRANT -fPIC

LIBS += $${DAE_CORE_LIB} \
        $${DAE_UNITS_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${DEALII_LIBS} \
        $${RT}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    fem_common.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    docstrings.h \
    fem_common.h \
    deal_ii.h \
    python_wraps.h

#######################################################
#                Install files
#######################################################
win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyDealII1.dll \
	$${PYDAE_DIR}/pyDealII.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
        $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
        $${PYDAE_DIR}/$${TARGET}.so
}
