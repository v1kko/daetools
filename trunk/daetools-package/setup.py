#!/usr/bin/env python
"""********************************************************************************
                               setup.py
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
********************************************************************************

Installation instructions

- GNU/Linux, MacOS (.tar.gz):
  python setup.py install

- Windows (.exe):
  python setup.py bdist_wininst --user-access-control force --install-script daetools_win_install.py --title "DAE Tools 1.6.0" --bitmap wininst.bmp

Create .tar.gz under GNU/Linux:
  python setup.py sdist --formats=gztar
  
"""

import os, sys, platform, shutil
from distutils.core import setup
from distutils import dir_util

daetools_version = '1.6.1'

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
    # So far there is no win64 port
    #if 'AMD64' in platform.machine():
    #    daetools_machine = 'win64'
    #else:
    #    daetools_machine = 'win32'
else:
    daetools_machine = str(platform.machine())

# (Platform/Python)-dependent shared libraries directory
# Now with removed compile-time dependency on numpy
platform_solib_dir = '{0}_{1}_py{2}{3}'.format(daetools_system, daetools_machine, python_major, python_minor)

shared_libs_dir = os.path.realpath('daetools/solibs')
shared_libs_dir = os.path.join(shared_libs_dir, '%s_%s' % (daetools_system, daetools_machine))

"""
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
"""
def _is_in_venv():
    return (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix or
            hasattr(sys, 'real_prefix'))

if _is_in_venv():
    inside_venv = True
else:
    inside_venv = False

if platform.system() == 'Linux':
    if not inside_venv:
        data_files = [
                        ('/usr/share/applications', [
                                                    'usr/share/applications/daetools-daeExamples.desktop',
                                                    'usr/share/applications/daetools-daePlotter.desktop'
                                                    ] ),
                        ('/usr/share/man/man1',     ['usr/share/man/man1/daetools.1.gz']),
                        ('/usr/share/menu',         [
                                                    'usr/share/menu/daetools-plotter',
                                                    'usr/share/menu/daetools-examples'
                                                    ] ),
                        ('/usr/share/pixmaps',      ['usr/share/pixmaps/daetools-48x48.png'])
                    ]
    else:
        data_files = []

    solibs = ['{0}/*.so'.format(platform_solib_dir)]
    fmi_solibs = 'fmi/{0}/*.so'.format(platform_solib_dir)

elif platform.system() == 'Windows':
    boost_python     = 'boost_python-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_python3    = 'boost_python3-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_system     = 'boost_system-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_thread     = 'boost_thread_win32-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_thread2    = 'boost_thread-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_filesystem = 'boost_filesystem-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_chrono     = 'boost_chrono-daetools-py{0}{1}'.format(python_major, python_minor)
    dae_config       = 'cdaeConfig-py{0}{1}'.format(python_major, python_minor)
    #deal_II          = 'deal_II-daetools'
    #sim_loader       = 'cdaeSimulationLoader-py{0}{1}'.format(python_major, python_minor)
    #fmu_so           = 'cdaeFMU_CS-py{0}{1}'.format(python_major, python_minor)
    mingw_dlls   = ['libgcc', 'libstdc++', 'libquadmath', 'libwinpthread', 'libgfortran', 'libssp']

    shared_libs = []

    print('shared_libs_dir = ', shared_libs_dir)
    if os.path.isdir(shared_libs_dir):
        shared_libs_files = os.listdir(shared_libs_dir)

        for f in shared_libs_files:
            if (boost_python in f) or (boost_python3 in f) or (boost_system in f) or (boost_thread in f) or (boost_thread2 in f) or (boost_filesystem in f) or (boost_chrono in f):
                shared_libs.append(os.path.join(shared_libs_dir, f))

            if dae_config in f:
                shared_libs.append(os.path.join(shared_libs_dir, f))

            #if (sim_loader in f) or (fmu_so in f):
            #    shared_libs.append(os.path.join(shared_libs_dir, f))

            for dll in mingw_dlls:
                if dll in f:
                    shared_libs.append(os.path.join(shared_libs_dir, f))
    #print('shared_libs = ', shared_libs)
    #raise RuntimeError('')

    for f in shared_libs:
        shutil.copy(f, 'daetools/pyDAE/{0}'.format(platform_solib_dir))
        shutil.copy(f, 'daetools/solvers/{0}'.format(platform_solib_dir))

    # Achtung!! data files dir must be '' in Windows
    data_files = []
    solibs = [
               '{0}/*.pyd'.format(platform_solib_dir),
               '{0}/*.dll'.format(platform_solib_dir)
             ]
    fmi_solibs = 'fmi/{0}/*.dll'.format(platform_solib_dir)

elif platform.system() == 'Darwin':
    if not inside_venv:
        data_files = [
                        #('/etc/daetools',           ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg']),
                        ('/usr/share/applications', [
                                                    'usr/share/applications/daetools-daeExamples.desktop',
                                                    'usr/share/applications/daetools-daePlotter.desktop'
                                                    ] ),
                        ('/usr/share/man/man1',     ['usr/share/man/man1/daetools.1.gz']),
                        ('/usr/share/menu',         [
                                                    'usr/share/menu/daetools-plotter',
                                                    'usr/share/menu/daetools-examples'
                                                    ] ),
                        ('/usr/share/pixmaps',      ['usr/share/pixmaps/daetools-48x48.png'])
                    ]
    else:
        data_files = []

    solibs = ['{0}/*.so'.format(platform_solib_dir)]
    fmi_solibs = 'fmi/{0}/*.dylib'.format(platform_solib_dir)

try:
    root_dir = os.path.dirname(os.path.abspath(__file__))
except:
    root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

docs_html_folder = os.path.join(root_dir, 'daetools', 'docs', 'html')
daetools_folder  = os.path.join(root_dir, 'daetools')

docs_html_dirs = [os.path.relpath(f[0], daetools_folder)+'/*.*' for f in os.walk(docs_html_folder)]
#print('\n'.join(docs_html_dirs))

setup(name = 'daetools',
      version = daetools_version,
      description = 'DAE Tools',
      long_description = 'Object-oriented equation-based modelling, simulation and optimisation software.',
      author = 'Dragan Nikolic',
      author_email = 'dnikolic@daetools.com',
      url = 'http://www.daetools.com',
      license = 'GNU GPL v3',
      packages = [
                   'daetools',
                   'daetools.pyDAE',
                   'daetools.solvers',
                   'daetools.solibs',
                   'daetools.code_generators',
                   'daetools.dae_plotter',
                   'daetools.dae_simulator',
                   'daetools.examples',
                   'daetools.unit_tests'
                 ],
      package_data = {
                       'daetools':                 ['*.txt',
                                                    '*.cfg',
                                                    'docs/presentations/*.pdf'
                                                   ] + docs_html_dirs,
                       'daetools.pyDAE':           solibs,
                       'daetools.solvers':         solibs,
                       'daetools.solibs':          ['%s_%s/*.*' % (daetools_system, daetools_machine)], # ?????
                       'daetools.dae_plotter':     ['images/*.png'],
                       'daetools.code_generators': ['c99/*.h', 'c99/*.c', 'c99/*.pro', 'c99/*.vcproj', 'c99/Makefile-*',
                                                    'cxx/*.h', 'cxx/*.cpp', 'cxx/*.pro', 'cxx/*.vcproj', 'cxx/Makefile-*',
                                                    fmi_solibs
                                                   ],
                       'daetools.dae_simulator':   ['images/*.png'],
                       'daetools.examples' :       ['*.pt', '*.init', '*.xsl', '*.css', '*.xml', '*.html', '*.sh', '*.bat', '*.png', 'meshes/*.msh', 'meshes/*.geo', 'meshes/*.png']
                     },
      data_files = data_files,
      scripts = ['scripts/create_shortcuts.js',
                 'scripts/daeplotter',
                 'scripts/daeexamples',
                 'scripts/daeplotter.bat',
                 'scripts/daeexamples.bat'],
      requires = ['numpy', 'scipy', 'matplotlib', 'PyQt5']
     )

if platform.system() == 'Windows':
    try:
        script_folder = os.path.dirname(os.path.abspath(__file__))
    except:
        script_folder = os.path.abspath(os.path.dirname(sys.argv[0]))
    script_folder    = os.path.join(script_folder, 'scripts')
    pythonw_path     = os.path.join(sys.prefix, 'pythonw.exe')
    python_version   = '%s.%s' % (python_major, python_minor)
    cmd = 'cscript %s\create_shortcuts.js %s %s %s' % (script_folder, pythonw_path, python_version, daetools_version)
    print('Creating shortcuts: %s ...' % cmd)
    os.system(cmd)

