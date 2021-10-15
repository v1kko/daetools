#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic
#************************************************************************************
# DAE Tools is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
#************************************************************************************
# Rename:
#  - libpyModule.so into pyModule.so
#  - pyModule.dll into pyModule.pyd
#  - libpyModule.dylib into pyModule.so
QMAKE_POST_LINK = $${MOVE_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT} \
                  $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

#dae_rename_module.commands = $${MOVE_FILE} \
#                                 $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                                 $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
#QMAKE_EXTRA_TARGETS += dae_rename_module

#message(py src: $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT})
#message(py dst: $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT})

# Install into daetools-dev
#dae_python_module_dev.depends += dae_rename_module
dae_python_module_dev.path     = $${DAE_INSTALL_PY_MODULES_DIR}
dae_python_module_dev.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Install into daetools-package
#dae_python_module_package.depends += dae_rename_module
dae_python_module_package.path     = $${SOLVERS_DIR}
dae_python_module_package.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# INSTALLS ignore rules if .files are not existing (can be resolved by using the target rule).
#target.depends += dae_rename_module
target.path     = $${DAE_DEST_DIR}
target.extra    = @echo Installing $${TARGET} # does nothing, overriding the default behaviour

INSTALLS += target \
            dae_python_module_dev \
            dae_python_module_package
