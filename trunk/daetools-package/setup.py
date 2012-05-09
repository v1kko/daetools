#!/usr/bin/env python 
import os, sys, platform
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

boost_solib_dir = os.path.realpath('../boost/stage/lib')
print boost_solib_dir

boost_solibs = []
if os.path.isdir(boost_solib_dir):
    boost_files = os.listdir(boost_solib_dir)
    boost_python = 'libboost_python-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_system = 'libboost_system-daetools-py{0}{1}'.format(python_major, python_minor)
    boost_thread = 'libboost_thread-daetools-py{0}{1}'.format(python_major, python_minor)

    for f in boost_files:
        if (boost_python in f) or (boost_system in f) or (boost_thread in f):
            boost_solibs.append(os.path.join(boost_solib_dir, f))  
        
print boost_solibs

if platform.system() == 'Linux':
    so_extension = 'so'
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
                 
elif platform.system() == 'Windows':
    so_extension = 'dll'
    data_files = [
                    ('c:/daetools',             ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg'])
                 ]
                 
elif platform.system() == 'Darwin':
    so_extension = 'so'
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

solibs   = ['{0}/*.{1}'.format(platform_solib_dir, so_extension)]


setup(name = 'daetools', 
      version = '1.2.0', 
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
 
