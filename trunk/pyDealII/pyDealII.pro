include(../dae.pri)
QT -= core gui
TARGET = pyDealII
TEMPLATE = lib
CONFIG += shared

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               #$${NUMPY_INCLUDE_DIR} \
               $${DEALII_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${BLAS_LAPACK_LIBDIR} \
                $${DEALII_LIB_DIR}

unix::QMAKE_CXXFLAGS += -fpermissive -fpic -Wall -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-unused-local-typedefs \
                        -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing -felide-constructors -Wno-unused \
                        -DBOOST_NO_HASH -DBOOST_NO_SLIST
unix::QMAKE_LFLAGS   += -pedantic -fpic -Wall -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-unused-local-typedefs \
                        -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing -felide-constructors -Wno-unused

win32-msvc2015::QMAKE_CXXFLAGS += -W2 -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing -felide-constructors \
                                  -DBOOST_NO_HASH -DBOOST_NO_SLIST
win32-msvc2015::QMAKE_LFLAGS   += /FORCE:MULTIPLE

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11
# deal.ii with msvc++ 2017 cannot use cxx11 (for some reasons)
win32-msvc2015::QMAKE_CXXFLAGS  +=
win32-msvc2015::QMAKE_LFLAGS    +=

LIBS += $${SOLIBS_RPATH}

LIBS += $${DEALII_LIBS} \
        $${DAE_CORE_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${RT}

SOURCES += stdafx.cpp \
           dllmain.cpp \
           dae_python.cpp

HEADERS += stdafx.h \
           docstrings.h \
           python_wraps.h

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_APPEND} \
                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
                  
# win32{
# QMAKE_POST_LINK = move /y \
# 	$${DAE_DEST_DIR}/pyDealII1.dll \
#     $${SOLVERS_DIR}/pyDealII.pyd
# }
# 
# unix{
# QMAKE_POST_LINK = cp -f \
#         $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
#         $${SOLVERS_DIR}/$${TARGET}.so
# }
