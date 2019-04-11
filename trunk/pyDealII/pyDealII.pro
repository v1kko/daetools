include(../dae.pri)
QT -= core gui

TARGET = pyDealII
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${DEALII_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${BLAS_LAPACK_LIBDIR} \
                $${DEALII_LIB_DIR}

unix::QMAKE_CXXFLAGS += -fpermissive -fpic -Wall -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-unused-local-typedefs \
                        -O2 -fstrict-aliasing -felide-constructors -Wno-unused -frounding-math -fsignaling-nans \
                        -DBOOST_NO_HASH -DBOOST_NO_SLIST
unix::QMAKE_LFLAGS   += -pedantic -fpic -Wall -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-unused-local-typedefs \
                        -O2 -fstrict-aliasing -felide-constructors -Wno-unused

win32-msvc2015::QMAKE_CXXFLAGS += -W2 -O2 -DBOOST_NO_HASH -DBOOST_NO_SLIST
# There are some multiply defined template functions (dealii::Vector<int>::lp_norm, dealii::Vector<class std::complex<float>>::operator=),
# use /FORCE:MULTIPLE to get around them.
win32-msvc2015::QMAKE_LFLAGS   += /FORCE:MULTIPLE

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11

#unix::QMAKE_CXXFLAGS  += --param ggc-min-expand=1

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
        $${BLAS_LAPACK_LIBS} \
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
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Rename libpyModule.so into pyModule.so
install_rename_module.commands = $${MOVE_FILE} \
                                 $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                                 $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
QMAKE_EXTRA_TARGETS += install_rename_module

# Install into daetools-dev
install_python_module.depends += install_rename_module
install_python_module.path     = $${DAE_INSTALL_PY_MODULES_DIR}
install_python_module.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Install into daetools-package
install_python_module2.depends += install_rename_module
install_python_module2.path     = $${SOLVERS_DIR}
install_python_module2.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# For some reasons INSTALLS was ignored without target rule.
target.depends += install_rename_module
target.path     = $${DAE_DEST_DIR}
target.extra    = @echo Installing $${TARGET} # do nothing, overriding the default behaviour

INSTALLS += target install_python_module install_python_module2
