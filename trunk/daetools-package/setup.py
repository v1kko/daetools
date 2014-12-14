#!/usr/bin/env python
"""********************************************************************************
                               setup.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2014
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************

Installation instructions

- GNU/Linux, MacOS (.tar.gz):
  python setup.py install

- Windows (.exe):
  python setup.py bdist_wininst --user-access-control force --install-script daetools_win_install.py --title "DAE Tools 1.4.0" --bitmap wininst.bmp

Create .tar.gz under GNU/Linux:
  python setup.py sdist --formats=gztar
  
"""

import os, sys, platform, shutil
from distutils.core import setup
from distutils.util import get_platform

# Python version
python_major = str(sys.version_info[0])
python_minor = str(sys.version_info[1])

# Numpy version (not required anymore)
#numpy_version = str(''.join(numpy.__version__.split('.')[0:2]))

# System := {'Linux', 'Windows', 'Darwin'}
daetools_system   = str(platform.system())

# Machine := {'i386', ..., 'i686', 'x86_64'}
if platform.system() == 'Darwin':
    daetools_machine = 'universal'
elif platform.system() == 'Windows':
    daetools_machine = 'win32'
else:
    daetools_machine = str(platform.machine())

# (Platform/Python)-dependent shared libraries directory
# Now with removed compile-time dependency on numpy
platform_solib_dir = '{0}_{1}_py{2}{3}'.format(daetools_system, daetools_machine, python_major, python_minor)
#print 'platform_solib_dir = ', platform_solib_dir

shared_libs_dir = os.path.realpath('solibs')
#print 'shared_libs_dir = ', shared_libs_dir

if daetools_machine == 'x86_64':
    if os.path.exists('/usr/lib') and os.path.exists('/usr/lib64'):
        # There are both /usr/lib and /usr/lib64
        usrlib = '/usr/lib64'
    elif os.path.exists('/usr/lib'):
        # There is only /usr/lib
        usrlib = '/usr/lib'
    elif os.path.exists('/usr/lib64'):
        # There is only /usr/lib64
        usrlib = '/usr/lib64'
    else:
        usrlib = '/usr/lib'
else:
    usrlib = '/usr/lib'

boost_python     = 'boost_python-daetools-py{0}{1}'.format(python_major, python_minor)
boost_system     = 'boost_system-daetools-py{0}{1}'.format(python_major, python_minor)
boost_thread     = 'boost_thread-daetools-py{0}{1}'.format(python_major, python_minor)
boost_filesystem = 'boost_filesystem-daetools-py{0}{1}'.format(python_major, python_minor)
deal_II          = 'deal_II-daetools'
sim_loader       = 'cdaeSimulationLoader'
if platform.system() == 'Windows':
    mingw_dlls   = ['libgcc', 'libstdc++', 'libquadmath', 'libwinpthread', 'libgfortran']
else:
    mingw_dlls   = []
    
shared_libs = []

if os.path.isdir(shared_libs_dir):
    boost_files = os.listdir(shared_libs_dir)

    for f in boost_files:
        if (boost_python in f) or (boost_system in f) or (boost_thread in f) or (boost_filesystem in f) or (deal_II in f) or (sim_loader in f):
            shared_libs.append(os.path.join(shared_libs_dir, f))

        for dll in mingw_dlls:
            if dll in f:
                shared_libs.append(os.path.join(shared_libs_dir, f))
    
#print 'shared_libs = ', shared_libs

if platform.system() == 'Linux':
    data_files = [
                    ('/etc/daetools',           ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg']),
                    ('/usr/share/applications', [
                                                'usr/share/applications/daetools-daeExamples.desktop',
                                                'usr/share/applications/daetools-daePlotter.desktop'
                                                ] ),
                    ('/usr/share/man/man1',     ['usr/share/man/man1/daetools.1.gz']),
                    ('/usr/share/menu',         [
                                                'usr/share/menu/daetools-plotter',
                                                'usr/share/menu/daetools-examples'
                                                ] ),
                    ('/usr/share/pixmaps',      ['usr/share/pixmaps/daetools-48x48.png']),
                    ('/usr/bin',                ['usr/bin/daeexamples']),
                    ('/usr/bin',                ['usr/bin/daeplotter']),
                    (usrlib,                    shared_libs)
                 ]
    solibs = ['{0}/*.so'.format(platform_solib_dir)]

elif platform.system() == 'Windows':
    for f in shared_libs:
        shutil.copy(f, 'daetools/pyDAE/{0}'.format(platform_solib_dir))
        shutil.copy(f, 'daetools/solvers/{0}'.format(platform_solib_dir))

    # Achtung!! data files dir must be '' in Windows
    data_files = [
                    ('', ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg']),
                    ('', [
                           'usr/share/applications/daeExamples.lnk',
                           'usr/share/applications/daePlotter.lnk'
                         ] ),
                 ]
    solibs = [
               '{0}/*.pyd'.format(platform_solib_dir),
               '{0}/*.dll'.format(platform_solib_dir)
             ]

elif platform.system() == 'Darwin':
    data_files = [
                    ('/etc/daetools',           ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg']),
                    ('/usr/share/applications', [
                                                'usr/share/applications/daetools-daeExamples.desktop',
                                                'usr/share/applications/daetools-daePlotter.desktop'
                                                ] ),
                    ('/usr/share/man/man1',     ['usr/share/man/man1/daetools.1.gz']),
                    ('/usr/share/menu',         [
                                                'usr/share/menu/daetools-plotter',
                                                'usr/share/menu/daetools-examples'
                                                ] ),
                    ('/usr/share/pixmaps',      ['usr/share/pixmaps/daetools-48x48.png']),
                    ('/usr/bin',                ['usr/bin/daeexamples']),
                    ('/usr/bin',                ['usr/bin/daeplotter']),
                    ('/usr/lib',                shared_libs)
                 ]

    solibs = ['{0}/*.so'.format(platform_solib_dir)]

#print 'solibs = ', solibs

setup(name = 'daetools',
      version = '1.4.0',
      description = 'DAE Tools',
      long_description = 'A cross-platform equation-oriented process modelling, simulation and optimization software (pyDAE modules).',
      author = 'Dragan Nikolic',
      author_email = 'dnikolic@daetools.com',
      url = 'http://www.daetools.com',
      license = 'GNU GPL v3',
      packages = [
                   'daetools',
                   'daetools.pyDAE',
                   'daetools.solvers',
                   'daetools.code_generators',
                   'daetools.dae_plotter',
                   'daetools.dae_simulator',
                   'daetools.examples',
                   'daetools.parsers',
                   'daetools.model_library',
                   'daetools.unit_tests'
                 ],
      package_data = {
                       'daetools':                 ['*.txt', 'docs/html/*.*',
                                                             'docs/html/_images/*.*',
                                                             'docs/html/_static/*.*',
                                                             'docs/html/_sources/*.*',
                                                             'docs/html/_modules/*.*',
                                                    'docs/latex/*.pdf',
                                                    'docs/presentations/*.pdf'
                                                   ],
                       'daetools.pyDAE':           solibs,
                       'daetools.solvers':         solibs,
                       'daetools.dae_plotter':     ['images/*.png'],
                       'daetools.code_generators': ['c99/*.h', 'c99/*.c', 'c99/*.pro', 'c99/*.vcproj', 'c99/Makefile-*', 'fmi/*daetools_fmi*.*'],
                       'daetools.dae_simulator':   ['images/*.png'],
                       'daetools.examples' :       ['*.init', '*.xsl', '*.css', '*.xml', '*.html', '*.sh', '*.bat', '*.png', 'meshes/*.msh', 'meshes/*.geo', 'meshes/*.png']
                     },
      data_files = data_files,
      scripts = ['scripts/daetools_win_install.py'],
      requires = ['numpy', 'scipy', 'matplotlib', 'PyQt4']
     )
