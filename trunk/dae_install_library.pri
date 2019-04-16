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
# Install headers and libs into daetools-dev
DAE_PROJECT_NAME = $$basename(PWD)

dae_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
dae_headers.files = *.h

dae_libs.path                   = $${DAE_INSTALL_LIBS_DIR}
dae_libs.files                  = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}
win32-msvc2015::dae_libs.files += $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${STATIC_LIB_EXT}

# Install into daetools-package
dae_py_solib.path  = $${SOLIBS_DIR}
dae_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# INSTALLS ignore rules if .files are not existing (can be resolved by using the target rule).
target.path  = $${DAE_DEST_DIR}
target.extra = @echo Installing $${TARGET} # does nothing, overriding the default behaviour

INSTALLS += target \
            dae_headers \
            dae_libs \
            dae_py_solib
