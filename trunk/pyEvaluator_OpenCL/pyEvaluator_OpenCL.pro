include(../dae.pri)
QT -= core
QT -= gui

TARGET = pyEvaluator_OpenCL
TEMPLATE = lib
CONFIG += shared plugin

# Debugging options
#QMAKE_CFLAGS += -g -O0
#QMAKE_CXXFLAGS += -g -O0

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

unix::LIBS += $${OPEN_CS_LIBS} \
              $${INTEL_OPENCL_LIBS} \
              $${BOOST_PYTHON_LIB} \
              $${BOOST_LIBS}

# Important: quotes around $${NVIDIA_OPENCL_LIBS} (in windows the path may include empty spaces).
win32-msvc2015::LIBS += $${OPEN_CS_LIBS} \
                        "$${NVIDIA_OPENCL_LIBS}" \
                        $${BOOST_PYTHON_LIB} \
                        $${BOOST_LIBS}

SOURCES += stdafx.cpp \
           dllmain.cpp \
           dae_python.cpp \

HEADERS += stdafx.h \
           resource.h

OTHER_FILES += Evaluator_OpenCL_resource.rc

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_APPEND} \
#                  $${PYDAE_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

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
