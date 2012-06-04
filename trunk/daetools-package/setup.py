#!/usr/bin/env python
import os, sys, platform, shutil
from distutils.core import setup
from distutils.util import get_platform

python_major = str(sys.version_info[0])
python_minor = str(sys.version_info[1])

# System := {'Linux', 'Windows', 'Darwin'}
daetools_system   = str(platform.system())

# Machine := {'i386', ..., 'i686', 'x86_64'}
daetools_machine  = str(platform.machine())

# (Platform/Python)-dependent shared libraries directory
platform_solib_dir = '{0}_{1}_py{2}{3}'.format(daetools_system, daetools_machine, python_major, python_minor)
print 'platform_solib_dir = ', platform_solib_dir

boost_solib_dir = os.path.realpath('solibs')
print 'boost_solib_dir = ', boost_solib_dir

boost_solibs = []
if os.path.isdir(boost_solib_dir):
    boost_files = os.listdir(boost_solib_dir)
    print 'boost_files = ', boost_files
    boost_python = 'boost_python-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_system = 'boost_system-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_thread = 'boost_thread-daetools-py{0}{1}'.format(python_major, python_minor)

    for f in boost_files:
        if (boost_python in f) or (boost_system in f) or (boost_thread in f):
            boost_solibs.append(os.path.join(boost_solib_dir, f))

print 'boost_solibs = ', boost_solibs

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
                    ('/usr/lib',                boost_solibs)
                 ]
    solibs = ['{0}/*.so'.format(platform_solib_dir)]

elif platform.system() == 'Windows':
    sys_drive = os.environ['SYSTEMDRIVE']
    config_dir = os.path.realpath( os.path.join(sys_drive, '\\daetools') )
    #print 'config_dir = ', config_dir

    for f in boost_solibs:
        shutil.copy(f, 'daetools/pyDAE/{0}'.format(platform_solib_dir))
        shutil.copy(f, 'daetools/solvers/{0}'.format(platform_solib_dir))

    data_files = [
                    (config_dir, ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg'])
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
                    ('/usr/lib',                boost_solibs)
                 ]

    solibs = ['{0}/*.so'.format(platform_solib_dir)]

print 'solibs = ', solibs

setup(name = 'daetools',
      version = '1.2.1',
      description = 'DAE Tools',
      long_description = 'A cross-platform equation-oriented process modelling, simulation and optimization software (pyDAE modules).',
      author = 'Dragan Nikolic',
      author_email = 'dnikolic@daetools.com',
      url = 'http://www.daetools.com',
      license = 'GNU GPL v3',
#     platforms = get_platform(),
      packages = [
                   'daetools',
                   'daetools.pyDAE',
                   'daetools.solvers',
                   'daetools.daePlotter',
                   'daetools.daeSimulator',
                   'daetools.examples',
                   'daetools.parsers',
                   'daetools.model_library'
                 ],
      package_data = {
                       'daetools':              ['*.txt', 'docs/*.html', 'docs/*.pdf'],
                       'daetools.pyDAE':        solibs,
                       'daetools.solvers':      solibs,
                       'daetools.daePlotter':   ['images/*.png'],
                       'daetools.daeSimulator': ['images/*.png'],
                       'daetools.examples' :    ['*.init', '*.xsl', '*.css', '*.xml', '*.html', '*.sh', '*.bat', '*.png']
                     },
      data_files = data_files,
      requires = ['numpy', 'scipy', 'matplotlib']
     )

