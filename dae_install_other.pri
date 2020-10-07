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
dae_headers_base.path  = $${DAE_INSTALL_HEADERS_DIR}
dae_headers_base.files = ../*.h

dae_headers_cape_open.path  = $${DAE_INSTALL_HEADERS_DIR}/CapeOpenThermoPackage
dae_headers_cape_open.files = ../CapeOpenThermoPackage/*.h

INSTALLS += dae_headers_base \
            dae_headers_cape_open
