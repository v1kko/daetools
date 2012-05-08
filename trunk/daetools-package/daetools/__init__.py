import os, sys, platform

# Python version
python_version_major = str(sys.version_info[0])
python_version_minor = str(sys.version_info[1])

# System := {'Linux', 'Windows', 'Darwin'}
daetools_system = str(platform.system())

# Machine := {'i386', ..., 'i686', 'AMD64'}
daetools_machine = str(platform.machine())

# daetools root directory
daetools_dir = os.path.join(os.path.dirname(__file__))

# pyDAE platform-dependant extension modules directory
pydae_sodir = os.path.join(daetools_dir, 'pyDAE', '{0}_{1}_py{2}{3}'.format(daetools_system, 
                                                                            daetools_machine, 
                                                                            python_version_major, 
                                                                            python_version_minor))
sys.path.append(pydae_sodir)

# solvers platform-dependant extension modules directory
solvers_sodir = os.path.join(daetools_dir, 'solvers', '{0}_{1}_py{2}{3}'.format(daetools_system, 
                                                                                daetools_machine, 
                                                                                python_version_major, 
                                                                                python_version_minor))
sys.path.append(solvers_sodir)

#print sys.path