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

- GNU/Linux, macOS, Windows (.tar.gz):
  python setup.py install

Create the source dist (.tar.gz):
  python setup.py sdist --formats=gztar
  
Create wheel:
  python setup.py bdist_wheel
"""

import os, sys, platform, shutil, pprint, re
from setuptools import setup, Distribution

daetools_version = '1.9.2'

# Python version
python_major = str(sys.version_info[0])
python_minor = str(sys.version_info[1])

daetools_system   = str(platform.system())

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

# (Platform/Python)-dependent shared libraries directory
# Now with removed compile-time dependency on numpy
so_lib_dir = '%s_%s/lib'    % (daetools_system, daetools_machine)
so_bin_dir = '%s_%s/bin'    % (daetools_system, daetools_machine)
py_mod_dir = '%s_%s/py%s%s' % (daetools_system, daetools_machine, python_major, python_minor)

def _is_in_venv():
    return (hasattr(sys, 'real_prefix') or 
           (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            sys.prefix.find('envs') != -1)

inside_venv = _is_in_venv()
print('inside_venv = %s' % ('True' if inside_venv else 'False'))

# install_name_tool needs to be run only once after compilation.
def install_name_tool(daetools_system, daetools_machine, python_major, python_minor):
    shared_libs_dir  = os.path.realpath('daetools')
    shared_libs_dir  = os.path.join(shared_libs_dir, 'solibs', '%s_%s' % (daetools_system, daetools_machine), 'lib')

    pymodules_dir = os.path.realpath('daetools')
    pymodules_dir = os.path.join(pymodules_dir, 'solibs', '%s_%s' % (daetools_system, daetools_machine), 'py%s%s' % (python_major, python_minor))

    shared_libs = []
    ext_modules = []
    
    if os.path.isdir(shared_libs_dir):
        shared_libs_files = os.listdir(shared_libs_dir)
        for f in shared_libs_files:
            shared_libs.append(os.path.join(shared_libs_dir, f))

    if os.path.isdir(pymodules_dir):
        ext_modules_files = os.listdir(pymodules_dir)
        for f in ext_modules_files:
            ext_modules.append(os.path.join(pymodules_dir, f))
    
    print('shared_libs_dir    = %s' % shared_libs_dir)
    print('pymodules_dir      = %s' % pymodules_dir)
    pprint.pprint(shared_libs)
    pprint.pprint(ext_modules)

    # Shared libraries' install_name must be @rpath/so_name so update their LC_ID_DYLIB section
    for so_path in shared_libs:
        dummy, so_name = os.path.split(so_path)
        cmd = 'install_name_tool -id @rpath/%s %s' % (so_name, so_path)
        #print(cmd)
        ret = os.system(cmd)
        print('%s returned [%s]' %(cmd,ret))

    # Python extension modules: 
    #  change the name of linked libraries from 'libName.dylib' to '@loader_path/../lib/libName.dylib'
    #  since they are in the ../lib directory.
    # Leave the libSystem and gcc libraries as they are.
    for ext_mod_path in ext_modules:
        dummy, ext_mod_name = os.path.split(ext_mod_path)
        print('Start install_name_tool -change for: %s' % ext_mod_name)
        for so_path in shared_libs:
            dummy, so_name = os.path.split(so_path)
            new_so_name = '@loader_path/../lib/%s' % so_name
            cmd = 'install_name_tool -change %s %s %s' % (so_name, new_so_name, ext_mod_path)
            #print(cmd)
            ret = os.system(cmd)
            print('    %s returned [%s]' %(cmd,ret))

    # Shared libraries: 
    #  change the name of linked libraries 'libName.dylib' to '@loader_path/libName.dylib' 
    #  since they are in the same directory.
    # Leave the libSystem and gcc libraries as they are.
    for so_lib_path in shared_libs:
        dummy, so_lib_name = os.path.split(so_lib_path)
        print('Start install_name_tool -change for: %s' % so_lib_name)
        for so_path in shared_libs:
            if so_path == so_lib_path: # skip itself
                continue            
            dummy, so_name = os.path.split(so_path)
            new_so_name = '@loader_path/%s' % so_name
            cmd = 'install_name_tool -change %s %s %s' % (so_name, new_so_name, so_lib_path)
            #print(cmd)
            ret = os.system(cmd)
            print('    %s returned [%s]' %(cmd,ret))
        print('')
    
    
    # pyDealII only:
    #  change the full path to the linked libdeal_II-daetools-8.5.0.dylib library to @loader_path/../lib/libdeal_II-daetools-8.5.0.dylib
    #  since fr some reasons it was linked using its full path .../trunk/deal.ii/build/lib.
    for so_lib_path in shared_libs:
        dummy, so_lib_name = os.path.split(so_lib_path)
        if deal_II in so_lib_name:
            print('Start install_name_tool -change for: %s' % so_lib_name)
            # Replace the full path to the libdeal_II-daetools in pyDealII.so
            dealii_daetools_lib = os.path.realpath('../deal.II/build/lib')
            dealii_daetools_lib = os.path.join(dealii_daetools_lib, so_lib_name)
            new_so_name = '@loader_path/../lib/%s' % so_lib_name
            pyDealII_path = os.path.join(pymodules_dir, 'pyDealII.so')
            cmd = 'install_name_tool -change %s %s %s' % (dealii_daetools_lib, new_so_name, pyDealII_path)
            #print(cmd)
            ret = os.system(cmd)
            print('    %s returned [%s]' %(cmd,ret))
    
boost_python     = 'boost_python-daetools-py{0}{1}'.format(python_major, python_minor)
boost_python3    = 'boost_python3-daetools-py{0}{1}'.format(python_major, python_minor)
boost_system     = 'boost_system-daetools-py{0}{1}'.format(python_major, python_minor)
boost_thread     = 'boost_thread_win32-daetools-py{0}{1}'.format(python_major, python_minor)
boost_thread2    = 'boost_thread-daetools-py{0}{1}'.format(python_major, python_minor)
boost_filesystem = 'boost_filesystem-daetools-py{0}{1}'.format(python_major, python_minor)
boost_chrono     = 'boost_chrono-daetools-py{0}{1}'.format(python_major, python_minor)
cdae_libs_py     = r"^cdae.*\-py{0}{1}.*".format(python_major, python_minor)
cdae_libs        = 'cdae'
vc_omp_lib       = 'vcomp'
opencs_libs      = 'OpenCS_'
deal_II          = 'deal_II-daetools'
sim_loader       = 'cdaeSimulationLoader-py{0}{1}'.format(python_major, python_minor)
fmu_so           = 'cdaeFMU_CS-py{0}{1}'.format(python_major, python_minor)
mingw_dlls   = ['libgcc', 'libstdc++', 'libquadmath', 'libwinpthread', 'libgfortran', 'libssp']
mac_gcc_libs = [] 
shared_libs  = []

if platform.system() == 'Linux':
    # Create only start menu links only for the current user
    if not inside_venv:
        data_files = [
                        (os.path.join(os.path.expanduser('~'), '.local/share/applications'), [
                                                                                                'usr/share/applications/daetools-daeExamples.desktop',
                                                                                                'usr/share/applications/daetools-daePlotter.desktop'
                                                                                             ] ),
                        (os.path.join(os.path.expanduser('~'), '.icons'),      ['usr/share/pixmaps/daetools-48x48.png'])
                    ]
    else:
        data_files = []

    solibs   = [ '%s/*.so*' % so_lib_dir ]
    pylibs   = [ '%s/*.so*' % py_mod_dir ]
    binaries = [ '%s/*'    % so_bin_dir ]
    
elif platform.system() == 'Windows':
    create_shortcuts_f = open(os.path.join('scripts', 'create_daetools_shortcuts.bat'), 'w')
    create_shortcuts_f.write('set CS_DIR=%~dp0\n')
    create_shortcuts_f.write('echo %CS_DIR%\n')
    create_shortcuts_f.write('cscript %%CS_DIR%%create_shortcuts.js %s %s.%s %s\n' % ('pythonw.exe', python_major, python_minor, daetools_version))
    create_shortcuts_f.close()
    
    # Achtung!! data files dir must be '' in Windows
    data_files = []
    solibs   = [ '%s/*.dll' % so_lib_dir ]
    pylibs   = [ '%s/*.pyd' % py_mod_dir ]
    binaries = [ '%s/*.exe' % so_bin_dir ]
    
elif platform.system() == 'Darwin':
    install_name_tool(daetools_system, daetools_machine, python_major, python_minor)

    data_files = []

    solibs   = [ '%s/*.dylib' % so_lib_dir ]
    pylibs   = [ '%s/*.so'    % py_mod_dir ]
    binaries = [ '%s/*'       % so_bin_dir ]
    install_name_tool(daetools_system, daetools_machine, python_major, python_minor)

try:
    root_dir = os.path.dirname(os.path.abspath(__file__))
except:
    root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

docs_html_folder = os.path.join(root_dir, 'daetools', 'docs', 'html')
daetools_folder  = os.path.join(root_dir, 'daetools')

docs_html_dirs = []
for root, dirs, files in os.walk(docs_html_folder):
    try:
        # Add "root/files[:]" to docs_html_dirs only if the files list is not empty.
        # This should resolve the error: [directory] doesn't exist or not a regular file,
        # which occurs in some systems where copying of the glob: "directory/*.*" fails
        # if there are no files in the directory (only folders).
        for f in files:
            full_path = os.path.join(root, f)
            relative_path = os.path.relpath(full_path, daetools_folder)
            docs_html_dirs.append(relative_path)
    except:
        pass
#print('\n'.join(docs_html_dirs))

####################################################################################
#                   setuptools.setup() function
####################################################################################
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

print(solibs + pylibs + binaries)

setup(name = 'daetools',
      version = daetools_version,
      description = 'DAE Tools',
      long_description = 'Object-oriented equation-based modelling, simulation and optimisation software',
      author = 'Dragan Nikolic',
      author_email = 'contact@daetools.com',
      url = 'http://www.daetools.com',
      license = 'GPLv3',
      packages = [
                   'daetools',
                   'daetools.pyDAE',
                   'daetools.solvers',
                   'daetools.solibs',
                   'daetools.code_generators',
                   'daetools.dae_plotter',
                   'daetools.dae_simulator',
                   'daetools.examples',
                   'daetools.unit_tests',
                   'daetools.ext_libs',
                   'daetools.ext_libs.pyevtk',
                   'daetools.ext_libs.SALib',
                   'daetools.ext_libs.SALib.analyze',
                   'daetools.ext_libs.SALib.plotting',
                   'daetools.ext_libs.SALib.sample',
                   'daetools.ext_libs.SALib.test_functions',
                   'daetools.ext_libs.SALib.util'
                 ],
      package_data = {
                       'daetools':                 ['*.txt',
                                                    '*.cfg',
                                                    'docs/presentations/*.pdf'
                                                   ] 
                                                   + docs_html_dirs, # <----------- DOCS
                       'daetools.solibs':          solibs      # platform_arch/lib 
                                                   + binaries  # platform_arch/bin
                                                   + pylibs,   # platform_arch/pyXY
                       'daetools.dae_plotter':     ['images/*.png'],
                       'daetools.code_generators': ['c99/*.h', 'c99/*.c', 'c99/*.pro', 'c99/*.vcproj', 'c99/Makefile-*',
                                                    'mpi/*.*', 
                                                    '*.css', '*.xsl'
                                                   ],
                       'daetools.dae_simulator':   ['*.html',
                                                    'images/*.*', 
                                                    'css/*.css',
                                                    'javascript/*.js',
                                                    'javascript/plotly/*.*',
                                                    'javascript/plotly/topojson/*.js'],
                       'daetools.examples' :       ['*.pt', '*.init', '*.xsl', '*.css', '*.xml', '*.html', '*.sh',  '*.c', '*.so', '*.csv',
                                                    '*.dylib', '*.dll', '*.bat', '*.png', 'meshes/*.msh', 'meshes/*.geo', 'meshes/*.png'],
                       'daetools.ext_libs.SALib':  ['*.txt'],
                       'daetools.ext_libs.pyevtk': ['*.txt']
                     },
      data_files = data_files,
      scripts = ['scripts/create_shortcuts.js',
                 'scripts/create_daetools_shortcuts.bat',
                 'scripts/daeplotter',
                 'scripts/daeexamples',
                 'scripts/daeplotter.bat',
                 'scripts/daeexamples.bat',
                 'scripts/daeplotter3',
                 'scripts/daeexamples3',
                 'scripts/daeplotter3.bat',
                 'scripts/daeexamples3.bat'],
      install_requires = ['numpy', 'scipy', 'matplotlib', 'lxml', 'pandas', 'openpyxl'],
      python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
      platforms = ['GNU/Linux', 'macOS', 'Windows'],
      classifiers = [ 'Development Status :: 5 - Production/Stable',
                      'Intended Audience :: Developers',
                      'Intended Audience :: Science/Research',
                      'Intended Audience :: Manufacturing',
                      'Topic :: Scientific/Engineering',
                      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                      'Operating System :: Microsoft :: Windows',
                      'Operating System :: MacOS',
                      'Operating System :: POSIX :: Linux',
                      'Programming Language :: Python :: 2',
                      'Programming Language :: Python :: 2.7',
                      'Programming Language :: Python :: 3',
                      'Programming Language :: Python :: 3.5',
                      'Programming Language :: Python :: 3.6',
                      'Programming Language :: Python :: 3.7',
                      'Programming Language :: Python :: 3.8',
                      'Programming Language :: Python :: 3.9'
                     ],
        keywords = 'modeling simulation optimization sensitivity_analysis parameter_estimation',
        
        # Set Pure-lib tag to False and indicate that the distribution being built contains extension modules.
        distclass = BinaryDistribution,
        cmdclass  = {'bdist_wheel': bdist_wheel}
     )

if platform.system() == 'Windows':
    try:
        script_folder = os.path.dirname(os.path.abspath(__file__))
    except:
        script_folder = os.path.abspath(os.path.dirname(sys.argv[0]))
    script_folder    = os.path.join(script_folder, 'scripts')
    pythonw_path     = os.path.join(sys.prefix, 'pythonw.exe')
    python_version   = '%s.%s' % (python_major, python_minor)
    cmd = 'cscript %s\create_shortcuts.js %s %s %s %s' % (script_folder, pythonw_path, python_version, daetools_version, daetools_machine)
    print('Creating shortcuts: %s ...' % cmd)
    os.system(cmd)
