"""********************************************************************************
                               __init__.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, sys, platform, numpy

# Fix for "IOError: [Errno 9] Bad file descriptor" when in pythonw.exe of Windows
if sys.platform.find('win') != -1 and sys.executable.find('pythonw') != -1:
    blackhole = open(os.devnull, 'w')
    sys.stdout = sys.stderr = blackhole
    
# Python version
python_version_major = str(sys.version_info[0])
python_version_minor = str(sys.version_info[1])

# Numpy version
numpy_version = str(''.join(numpy.__version__.split('.')[0:2]))

# System := {'Linux', 'Windows', 'Darwin'}
daetools_system = str(platform.system())

# Machine := {'i386', ..., 'i686', 'x86_64'}
if platform.system() == 'Darwin':
    daetools_machine = str(platform.machine())
elif platform.system() == 'Windows':
    if platform.architecture()[0] == '64bit':
        daetools_machine = 'win64'
    else:
        daetools_machine = 'win32'
else:
    daetools_machine = str(platform.machine())

# daetools root directory
daetools_dir = os.path.dirname(os.path.realpath(__file__))

# pyDAE platform-dependant extension modules directory
# Now with removed compile-time dependency on numpy
pydae_sodir = os.path.join(daetools_dir, 'pyDAE', '{0}_{1}_py{2}{3}'.format(daetools_system,
                                                                            daetools_machine,
                                                                            python_version_major,
                                                                            python_version_minor))
sys.path.append(pydae_sodir)

# solvers platform-dependant extension modules directory
# Now with removed compile-time dependency on numpy
solvers_sodir = os.path.join(daetools_dir, 'solvers', '{0}_{1}_py{2}{3}'.format(daetools_system,
                                                                                daetools_machine,
                                                                                python_version_major,
                                                                                python_version_minor))
sys.path.append(solvers_sodir)

# solibs platform-dependant extension modules directory
solibs_sodir = os.path.join(daetools_dir, 'solibs', '{0}_{1}'.format(daetools_system,
                                                                     daetools_machine))
sys.path.append(solibs_sodir)

#print sys.path
