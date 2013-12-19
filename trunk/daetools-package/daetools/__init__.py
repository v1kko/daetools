import os, sys, platform, numpy

# Python version
python_version_major = str(sys.version_info[0])
python_version_minor = str(sys.version_info[1])

# Numpy version
numpy_version = str(''.join(numpy.__version__.split('.')[0:2]))

# System := {'Linux', 'Windows', 'Darwin'}
daetools_system = str(platform.system())

# Machine := {'i386', ..., 'i686', 'x86_64'}
if platform.system() == 'Darwin':
    daetools_machine = 'universal'
elif platform.system() == 'Windows':
    daetools_machine = 'win32'
else:
    daetools_machine = str(platform.machine())

# daetools root directory
daetools_dir = os.path.join(os.path.dirname(__file__))

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

#print sys.path