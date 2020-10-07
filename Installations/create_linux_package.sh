#!/bin/bash

set -e

case $1 in
  -h* ) echo "Usage:"
        echo "create_linux_package.sh"
        echo " "
        return
        ;;
esac

INSTALLATIONS_DIR="$( cd "$( dirname "$0" )" && pwd )"
# ACHTUNG! cd to INSTALLATIONS_DIR (in case the script is called from some other folder)
echo ${INSTALLATIONS_DIR}
cd ${INSTALLATIONS_DIR}

# if [ "$1" = "" ]; then
#   PYTHON="python"
# else
#   PYTHON="$1"
# fi

PYTHON="python"

VER_MAJOR=
VER_MINOR=
VER_BUILD=
PACKAGE_NAME=daetools
SOURCE_FOLDER="${INSTALLATIONS_DIR}/../daetools"
PCKG_TYPE=
ARCH=
LIB=
ARCH_RPM=
RELEASE_DIR=${INSTALLATIONS_DIR}/../release
HOST_ARCH=`uname -m`
PLATFORM=`uname -s | tr "[:upper:]" "[:lower:]"`

if [ ${HOST_ARCH} = "x86_64" ]; then
  LIB=lib64
  ARCH=amd64
  ARCH_RPM=x86_64
elif [ ${HOST_ARCH} = "armv5tejl" ]; then
  LIB=lib
  ARCH=armel
  ARCH_RPM=armel
elif [ ${HOST_ARCH} = "i386" ]; then
  LIB=lib
  ARCH=i386
  ARCH_RPM=i386
elif [ ${HOST_ARCH} = "i586" ]; then
  LIB=lib
  ARCH=i386
  ARCH_RPM=i386
elif [ ${HOST_ARCH} = "i686" ]; then
  LIB=lib
  ARCH=i386
  ARCH_RPM=i386
else
  echo "ERROR: undefined architecture"
  exit
fi

if [ ${PLATFORM} = "linux" ]; then
  SO="so"
  DISTRIBUTOR_ID=`${PYTHON} -c "import platform; print(platform.linux_distribution()[0])" | tr "[:upper:]" "[:lower:]"`
  CODENAME=`${PYTHON} -c "import platform; print(platform.linux_distribution()[1])" | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}-${CODENAME}
  echo $DISTRO

elif [ ${PLATFORM} = "gnu/kfreebsd" ]; then
  SO="so"
  ARCH=kfreebsd-${ARCH}
  DISTRIBUTOR_ID=`echo $(lsb_release -si) | tr "[:upper:]" "[:lower:]"`
  CODENAME=`echo $(lsb_release -sc) | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}-${CODENAME}

elif [ ${PLATFORM} = "darwin" ]; then
  SO="dylib"
  DISTRIBUTOR_ID="macosx"
  CODENAME=`echo $(sw_vers -productVersion) | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}

else
  echo "ERROR: undefined platform: ${PLATFORM}"
  exit
fi
 
COMPILER_SETTINGS_DIR="${INSTALLATIONS_DIR}/../.compiler_settings"

PYTHON_MAJOR=`${PYTHON} -c "import sys; print(sys.version_info[0])"`
PYTHON_MINOR=`${PYTHON} -c "import sys; print(sys.version_info[1])"`
PYTHON_VERSION=${PYTHON_MAJOR}.${PYTHON_MINOR}
VER_MAJOR=`cat ${COMPILER_SETTINGS_DIR}/dae_major`
VER_MINOR=`cat ${COMPILER_SETTINGS_DIR}/dae_minor`
VER_BUILD=`cat ${COMPILER_SETTINGS_DIR}/dae_build`
VERSION=${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}
#SITE_PACKAGES_DIR=`${PYTHON} -c "import distutils.sysconfig; print (distutils.sysconfig.get_python_lib())"`

# PYTHON
#PYTHON_TYPE=`cat ${COMPILER_SETTINGS_DIR}/python`

# BOOST
BOOST_BUILD_TYPE=`cat ${COMPILER_SETTINGS_DIR}/boost`
if [ ${BOOST_BUILD_TYPE} = "custom" ]; then 
  BOOST_PYTHON_LIB_NAME=boost_python-daetools-py${PYTHON_MAJOR}${PYTHON_MINOR}
  BOOST_PYTHON_LIB=lib${BOOST_PYTHON_LIB_NAME}.${SO}
  BOOST_SYSTEM_LIB_NAME=boost_system-daetools-py${PYTHON_MAJOR}${PYTHON_MINOR}
  BOOST_SYSTEM_LIB=lib${BOOST_SYSTEM_LIB_NAME}.${SO}
  BOOST_THREAD_LIB_NAME=boost_thread-daetools-py${PYTHON_MAJOR}${PYTHON_MINOR}
  BOOST_THREAD_LIB=lib${BOOST_THREAD_LIB_NAME}.${SO}
fi

echo $BOOST_BUILD_TYPE
echo $BOOST_PYTHON_LIB
echo $BOOST_SYSTEM_LIB
echo $BOOST_THREAD_LIB

IDAS=../idas/build
BONMIN=../bonmin/build
NLOPT=../nlopt
SUPERLU=../superlu
SUPERLU_MT=../superlu_mt
MAGMA=../magma
TRILINOS=../trilinos/build

if [ ! -n ${VER_MAJOR} ]; then
  echo "Invalid daetools version major number"
  return
fi
if [ ! -n ${VER_MINOR} ]; then
  echo "Invalid daetools version minor number"
  return
fi
if [ ! -n ${VER_BUILD} ]; then
  echo "Invalid daetools version build number"
  return
fi

USRLIB=/usr/${LIB}

if [ "$1" = "cdae" ]; then
  PCKG_TYPE="tgz"

elif [ ${DISTRIBUTOR_ID} = "debian" ]; then
  PCKG_TYPE="deb"

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  PCKG_TYPE="deb"

elif [ ${DISTRIBUTOR_ID} = "linuxmint" ]; then
  PCKG_TYPE="deb"

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  PCKG_TYPE="rpm"

elif [ ${DISTRIBUTOR_ID} = "centos" ]; then
  PCKG_TYPE="rpm"

elif [ ${DISTRIBUTOR_ID} = "macosx" ]; then
  PCKG_TYPE="distutils.tar.gz"

else
  PCKG_TYPE="distutils.tar.gz"
fi

TGZ=${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${ARCH}.tar.gz
RPM=${PACKAGE_NAME}-${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}.${ARCH_RPM}.rpm
if [ ${PLATFORM} = "darwin" ]; then
  BUILD_DIR=${INSTALLATIONS_DIR}/${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${DISTRO}
else
  BUILD_DIR=${INSTALLATIONS_DIR}/${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${ARCH}_${DISTRO}
fi
USRBIN=${BUILD_DIR}/usr/bin

echo " "
echo " The package settings:"
echo "    version:           " ${VERSION}
echo "    platform:          " ${PLATFORM}
echo "    os:                " ${DISTRO}
echo "    package type:      " ${PCKG_TYPE}
echo "    host architecture: " ${HOST_ARCH}
echo "    architecture:      " ${ARCH}
echo "    lib prefix:        " ${LIB}
echo "    python version:    " ${PYTHON_VERSION}
echo "    boost version:     " ${BOOST_VERSION}
echo "    /usr/lib dir:      " ${USRLIB}
echo "    build dir:         " ${BUILD_DIR}
echo " " 
read -p " Proceed [y/n]? " do_proceed
case ${do_proceed} in
  [Nn]* ) echo "Aborting ..."
          exit;;
      * ) break;;
esac

# if [ ${PCKG_TYPE} = "tgz" ]; then
#   INSTALL_DIR=${PACKAGE_NAME}-${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}
#   mkdir ${INSTALL_DIR}
#   mkdir ${INSTALL_DIR}/lib
#   mkdir ${INSTALL_DIR}/include
#   mkdir ${INSTALL_DIR}/include/Core
#   mkdir ${INSTALL_DIR}/include/Activity
#   mkdir ${INSTALL_DIR}/include/DataReporting
#   mkdir ${INSTALL_DIR}/include/IDAS_DAESolver
#   mkdir ${INSTALL_DIR}/include/BONMIN_MINLPSolver
#   mkdir ${INSTALL_DIR}/include/LA_SuperLU
#   mkdir ${INSTALL_DIR}/TestPrograms
# 
#   cp ../compile_libraries_linux.sh            ${INSTALL_DIR}
# 
#   cp ../release/libcdaeActivity.a             ${INSTALL_DIR}/lib
#   cp ../release/libcdaeBONMIN_MINLPSolver.a   ${INSTALL_DIR}/lib
#   cp ../release/libcdaeCore.a                 ${INSTALL_DIR}/lib
#   cp ../release/libcdaeDataReporting.a        ${INSTALL_DIR}/lib
#   cp ../release/libcdaeIDAS_DAESolver.a       ${INSTALL_DIR}/lib
#   cp ../release/libcdaeIPOPT_NLPSolver.a      ${INSTALL_DIR}/lib
#   cp ../release/libcdaeNLOPT_NLPSolver.a      ${INSTALL_DIR}/lib
#   cp ../release/libcdaeSuperLU_LASolver.a     ${INSTALL_DIR}/lib
#   cp ../release/libcdaeSuperLU_MT_LASolver.a  ${INSTALL_DIR}/lib
# 
#   cp ../dae.h                                           ${INSTALL_DIR}/include
#   cp ../dae_develop.h                                   ${INSTALL_DIR}/include
#   cp ../config.h                                        ${INSTALL_DIR}/include
#   cp ../Core/definitions.h                              ${INSTALL_DIR}/include/Core
#   cp ../Core/xmlfile.h                                  ${INSTALL_DIR}/include/Core
#   cp ../Core/coreimpl.h                                 ${INSTALL_DIR}/include/Core
#   cp ../Core/helpers.h                                  ${INSTALL_DIR}/include/Core
#   cp ../Core/base_logging.h                             ${INSTALL_DIR}/include/Core
#   cp ../Core/class_factory.h                            ${INSTALL_DIR}/include/Core
#   cp ../Activity/base_activities.h                      ${INSTALL_DIR}/include/Activity
#   cp ../Activity/simulation.h                           ${INSTALL_DIR}/include/Activity
#   cp ../DataReporting/datareporters.h                   ${INSTALL_DIR}/include/DataReporting
#   cp ../DataReporting/base_data_reporters_receivers.h   ${INSTALL_DIR}/include/DataReporting
#   cp ../IDAS_DAESolver/base_solvers.h                   ${INSTALL_DIR}/include/IDAS_DAESolver
#   cp ../IDAS_DAESolver/ida_solver.h                     ${INSTALL_DIR}/include/IDAS_DAESolver
#   cp ../BONMIN_MINLPSolver/base_solvers.h               ${INSTALL_DIR}/include/BONMIN_MINLPSolver
#   cp ../LA_SuperLU/superlu_solvers.h                    ${INSTALL_DIR}/include/LA_SuperLU
#   
#   cp ../dae.pri                                         ${INSTALL_DIR}
#   cp ../TestPrograms/TestPrograms.pro                   ${INSTALL_DIR}/TestPrograms
#   cp ../TestPrograms/main.cpp                           ${INSTALL_DIR}/TestPrograms
#   cp ../TestPrograms/*tutorial*.h                       ${INSTALL_DIR}/TestPrograms
#   cp ../TestPrograms/whats_the_time.h                   ${INSTALL_DIR}/TestPrograms
#   cp ../TestPrograms/variable_types.h                   ${INSTALL_DIR}/TestPrograms
# 
#   tar -czvf ${TGZ} ${INSTALL_DIR}
#   rm -r ${INSTALL_DIR}
#   exit
# fi

# if [ -d ${PACKAGE_NAME} ]; then
#   rm -r ${PACKAGE_NAME}
# fi
# mkdir ${PACKAGE_NAME}
# mkdir ${PACKAGE_NAME}/docs
# mkdir ${PACKAGE_NAME}/docs/images
# mkdir ${PACKAGE_NAME}/docs/api_ref
# mkdir ${PACKAGE_NAME}/daePlotter
# mkdir ${PACKAGE_NAME}/daePlotter/images
# mkdir ${PACKAGE_NAME}/examples
# mkdir ${PACKAGE_NAME}/examples/images
# mkdir ${PACKAGE_NAME}/pyDAE
# mkdir ${PACKAGE_NAME}/daeSimulator
# mkdir ${PACKAGE_NAME}/daeSimulator/images
# mkdir ${PACKAGE_NAME}/solvers
# mkdir ${PACKAGE_NAME}/model_library

if [ -d ${BUILD_DIR} ]; then
  rm -r ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
#mkdir ${BUILD_DIR}/usr
#mkdir ${BUILD_DIR}/usr/bin
#mkdir ${BUILD_DIR}${USRLIB}

# 
# # Python extension modules and LA solvers
# cp ../release/pyCore.so             ${PACKAGE_NAME}/pyDAE
# cp ../release/pyActivity.so         ${PACKAGE_NAME}/pyDAE
# cp ../release/pyDataReporting.so    ${PACKAGE_NAME}/pyDAE
# cp ../release/pyIDAS.so             ${PACKAGE_NAME}/pyDAE
# cp ../release/pyUnits.so            ${PACKAGE_NAME}/pyDAE
# 
# if [ -e ../release/pyBONMIN.so ]; then
#   cp ../release/pyBONMIN.so          ${PACKAGE_NAME}/solvers
# fi
# 
# if [ -e ../release/pyIPOPT.so ]; then
#   cp ../release/pyIPOPT.so          ${PACKAGE_NAME}/solvers
# fi
# 
# if [ -e ../release/pyNLOPT.so ]; then
#   cp ../release/pyNLOPT.so          ${PACKAGE_NAME}/solvers
# fi
# 
# #if [ -e ../release/pyAmdACML.so ]; then
# #  cp ../release/pyAmdACML.so          ${PACKAGE_NAME}/solvers
# #fi
# 
# #if [ -e ../release/pyIntelMKL.so ]; then
# #  cp ../release/pyIntelMKL.so         ${PACKAGE_NAME}/solvers
# #fi
# 
# #if [ -e ../release/pyLapack.so ]; then
# #  cp ../release/pyLapack.so           ${PACKAGE_NAME}/solvers
# #fi
# 
# #if [ -e ../release/pyMagma.so ]; then
# #  cp ../release/pyMagma.so             ${PACKAGE_NAME}/solvers
# #fi
# 
# #if [ -e ../release/pyCUSP.so ]; then
# #  cp ../release/pyCUSP.so              ${PACKAGE_NAME}/solvers
# #fi
# 
# if [ -e ../release/pySuperLU.so ]; then
#   cp ../release/pySuperLU.so           ${PACKAGE_NAME}/solvers
# fi
# if [ -e ../release/pySuperLU_MT.so ]; then
#   cp ../release/pySuperLU_MT.so        ${PACKAGE_NAME}/solvers
# fi
# if [ -e ../release/pySuperLU_CUDA.so ]; then
#   cp ../release/pySuperLU_CUDA.so      ${PACKAGE_NAME}/solvers
# fi
# 
# #if [ -e ../release/pyIntelPardiso.so ]; then
# #  cp ../release/pyIntelPardiso.so     ${PACKAGE_NAME}/solvers
# #fi
# 
# if [ -e ../release/pyTrilinos.so ]; then
#   cp ../release/pyTrilinos.so   ${PACKAGE_NAME}/solvers
# fi
# 
# # Licences
# cp ../licence*                                   ${PACKAGE_NAME}
# cp ../ReadMe.txt                                 ${PACKAGE_NAME}
# 
# # Python files
# cp ../python-files/daetools__init__.py           ${PACKAGE_NAME}/__init__.py
# cp ../python-files/daeLogs.py                    ${PACKAGE_NAME}/pyDAE
# cp ../python-files/WebView_ui.py                 ${PACKAGE_NAME}/pyDAE
# cp ../python-files/WebViewDialog.py              ${PACKAGE_NAME}/pyDAE
# cp ../python-files/daeLogs.py                    ${PACKAGE_NAME}/pyDAE
# cp ../python-files/daeVariableTypes.py           ${PACKAGE_NAME}/pyDAE
# cp ../python-files/daeDataReporters.py           ${PACKAGE_NAME}/pyDAE
# cp ../python-files/pyDAE__init__.py              ${PACKAGE_NAME}/pyDAE/__init__.py
# cp ../python-files/solvers__init__.py            ${PACKAGE_NAME}/solvers/__init__.py
# cp ../python-files/aztecoo_options.py            ${PACKAGE_NAME}/solvers
# cp ../python-files/daeMinpackLeastSq.py          ${PACKAGE_NAME}/solvers
# cp ../python-files/model_library__init__.py      ${PACKAGE_NAME}/model_library/__init__.py
# 
# # daeSimulator
# cp ../python-files/daeSimulator/__init__.py      ${PACKAGE_NAME}/daeSimulator
# cp ../python-files/daeSimulator/daeSimulator.py  ${PACKAGE_NAME}/daeSimulator
# cp ../python-files/daeSimulator/Simulator_ui.py  ${PACKAGE_NAME}/daeSimulator
# cp ../python-files/daeSimulator/images/*.*       ${PACKAGE_NAME}/daeSimulator/images
# 
# # daePlotter
# cp ../python-files/daePlotter/*.py               ${PACKAGE_NAME}/daePlotter
# cp ../python-files/daePlotter/images/*.*         ${PACKAGE_NAME}/daePlotter/images
# 
# # Model Library
# cp ../python-files/model_library/*.py            ${PACKAGE_NAME}/model_library
# 
# # Examples and Tutorials
# cp ../python-files/examples/*.css                ${PACKAGE_NAME}/examples
# cp ../python-files/examples/*.xsl                ${PACKAGE_NAME}/examples
# cp ../python-files/examples/__init__.py          ${PACKAGE_NAME}/examples
# cp ../python-files/examples/*tutorial*.*         ${PACKAGE_NAME}/examples
# cp ../python-files/examples/*RunExamples*.py     ${PACKAGE_NAME}/examples
# cp ../python-files/examples/*whats_the_time*.*   ${PACKAGE_NAME}/examples
# cp ../python-files/examples/*.init               ${PACKAGE_NAME}/examples
# cp ../python-files/examples/images/*.*           ${PACKAGE_NAME}/examples/images
# 
# # Documentation
# cp ../python-files/api_ref/*.html  ${PACKAGE_NAME}/docs/api_ref
# 
# # Strip python extension modules
# find ${PACKAGE_NAME}/pyDAE   -name \*.so* | xargs strip -S
# find ${PACKAGE_NAME}/solvers -name \*.so* | xargs strip -S
# 
# # Config
# mkdir -p ${PACKAGE_NAME}/etc/daetools
# cp ../daetools.cfg  ${PACKAGE_NAME}/etc/daetools
# cp ../bonmin.cfg    ${PACKAGE_NAME}/etc/daetools
# chmod go-wx ${PACKAGE_NAME}/etc/daetools/daetools.cfg
# chmod go-wx ${PACKAGE_NAME}/etc/daetools/bonmin.cfg

# BOOST
#if [${BOOST_BUILD_TYPE} = "custom" ]; then
#  mkdir -p ${PACKAGE_NAME}/usr/lib
#  cp ${BOOST_LIB_DIR}/${BOOST_PYTHON_LIB} ${PACKAGE_NAME}/usr/lib
#  cp ${BOOST_LIB_DIR}/${BOOST_SYSTEM_LIB} ${PACKAGE_NAME}/usr/lib
#  cp ${BOOST_LIB_DIR}/${BOOST_THREAD_LIB} ${PACKAGE_NAME}/usr/lib
#fi

# daePlotter and daeRunExamples
# mkdir -p ${PACKAGE_NAME}/usr/bin
# echo "#!/bin/sh"                                                                            > ${PACKAGE_NAME}/usr/bin/daeplotter
# echo "${PYTHON} -c \"from daetools.daePlotter import daeStartPlotter; daeStartPlotter()\"" >> ${PACKAGE_NAME}/usr/bin/daeplotter
# chmod +x ${PACKAGE_NAME}/usr/bin/daeplotter
# 
# echo "#!/bin/sh"                                                                                              > ${PACKAGE_NAME}/usr/bin/daeexamples
# echo "${PYTHON} -c \"from daetools.examples.python.daeRunExamples import daeRunExamples; daeRunExamples()\"" >> ${PACKAGE_NAME}/usr/bin/daeexamples
# chmod +x ${PACKAGE_NAME}/usr/bin/daeexamples

# if [ ${PLATFORM} = "darwin" ]; then
#   DAE_PLOTTER=/Applications/daetools/daePlotter.app/Contents/MacOS
#   DAE_EXAMPLES=/Applications/daetools/daeExamples.app/Contents/MacOS
# else
#   DAE_PLOTTER=/usr/bin
#   DAE_EXAMPLES=/usr/bin
# fi

# SETUP_PY=setup.py
# echo "#!/usr/bin/env python " > ${SETUP_PY}
# echo "import sys " >> ${SETUP_PY}
# echo "from distutils.core import setup " >> ${SETUP_PY}
# echo " " >> ${SETUP_PY}
# echo "setup(name='${PACKAGE_NAME}', " >> ${SETUP_PY}
# echo "      version='${VERSION}', " >> ${SETUP_PY}
# echo "      description='DAE Tools', " >> ${SETUP_PY}
# echo "      long_description='A cross-platform equation-oriented process modelling software (pyDAE modules).', " >> ${SETUP_PY}
# echo "      author='Dragan Nikolic', " >> ${SETUP_PY}
# echo "      author_email='dnikolic@daetools.com', " >> ${SETUP_PY}
# echo "      url='http://www.daetools.com', " >> ${SETUP_PY}
# echo "      license='GNU GPL v3', " >> ${SETUP_PY}
# echo "      platforms='${ARCH}', " >> ${SETUP_PY}
# echo "      packages=['${PACKAGE_NAME}'], " >> ${SETUP_PY}
# echo "      package_dir={'${PACKAGE_NAME}': '${SOURCE_FOLDER}'}, " >> ${SETUP_PY}
# echo "      package_data={'${PACKAGE_NAME}': ['*.*', 'pyDAE/*.so', 'pyDAE/*.py', 'solvers/*.so', 'solvers/*.py', 'model_library/*.py', 'examples/*.py', 'examples/python/*.*', 'examples/c++/*.*', 'parsers/*.py', 'docs/*.html', 'docs/*.pdf', 'daeSimulator/*.py', 'daeSimulator/images/*.png', 'daePlotter/*.py', 'daePlotter/images/*.png']}, " >> ${SETUP_PY}
# echo "      data_files=[('/etc/daetools', ['${SOURCE_FOLDER}/etc/daetools/daetools.cfg', '${SOURCE_FOLDER}/etc/daetools/bonmin.cfg'] ), " >> ${SETUP_PY}
# #if [${BOOST_BUILD_TYPE} = "custom" ]; then
# #echo "                  ('${USRLIB}', ['${PACKAGE_NAME}/usr/lib/${BOOST_PYTHON_LIB}', '${PACKAGE_NAME}/usr/lib/${BOOST_SYSTEM_LIB}', '${PACKAGE_NAME}/usr/lib/${BOOST_THREAD_LIB}']), "  >> ${SETUP_PY}
# #fi
# echo "                  ('/usr/share/applications', ['${SOURCE_FOLDER}/usr/share/applications/daetools-daeExamples.desktop', '${SOURCE_FOLDER}/usr/share/applications/daetools-daePlotter.desktop'] ), " >> ${SETUP_PY}
# echo "                  ('/usr/share/man/man1',     ['${SOURCE_FOLDER}/usr/share/man/man1/daetools.1.gz'] ), "                                                       >> ${SETUP_PY}
# echo "                  ('/usr/share/menu',         ['${SOURCE_FOLDER}/usr/share/menu/daetools-plotter', '${SOURCE_FOLDER}/usr/share/menu/daetools-examples'] ), "   >> ${SETUP_PY}
# echo "                  ('/usr/share/pixmaps',      ['${SOURCE_FOLDER}/usr/share/pixmaps/daetools_main.png'] ), "                                                    >> ${SETUP_PY}
# echo "                  ('/usr/bin',   ['${SOURCE_FOLDER}/usr/bin/daeexamples'] ), " >> ${SETUP_PY}
# echo "                  ('/usr/bin',   ['${SOURCE_FOLDER}/usr/bin/daeplotter'] ) ]"  >> ${SETUP_PY}
# echo "      ) " >> ${SETUP_PY}
# echo " " >> ${SETUP_PY}

# echo "#!/usr/bin/env python 
# import sys 
# from distutils.core import setup 
# from distutils.util import get_platform
# 
# setup(name='daetools', 
#       version='${VERSION}', 
#       description='DAE Tools', 
#       long_description='A cross-platform equation-oriented process modelling software (pyDAE modules).', 
#       author='Dragan Nikolic', 
#       author_email='dnikolic@daetools.com', 
#       url='http://www.daetools.com', 
#       license='GNU GPL v3', 
# #     platforms = get_platform(),
#       packages=['daetools'], 
#       package_dir={'daetools': '.'}, 
#       package_data={'daetools': ['__init__.py', '*.txt', 'pyDAE/*.so', 'pyDAE/*.py', 'solvers/*.so', 'solvers/*.py', 'model_library/*.py', 'examples/*.py', 'examples/python/*.*', 'examples/c++/*.*', 'parsers/*.py', 'docs/*.html', 'docs/*.pdf', 'daeSimulator/*.py', 'daeSimulator/images/*.png', 'daePlotter/*.py', 'daePlotter/images/*.png']}, 
#       data_files=[('/etc/daetools',           ['etc/daetools/daetools.cfg', 'etc/daetools/bonmin.cfg'] ), 
#                   ('/usr/share/applications', ['usr/share/applications/daetools-daeExamples.desktop', 'usr/share/applications/daetools-daePlotter.desktop'] ), 
#                   ('/usr/share/man/man1',     ['usr/share/man/man1/daetools.1.gz'] ), 
#                   ('/usr/share/menu',         ['usr/share/menu/daetools-plotter', 'usr/share/menu/daetools-examples'] ), 
#                   ('/usr/share/pixmaps',      ['usr/share/pixmaps/daetools_main.png'] ), 
#                   ('/usr/bin',                ['usr/bin/daeexamples'] ), 
#                   ('/usr/bin',                ['usr/bin/daeplotter'] ) ]
#       )" > setup.py

if [ ${PCKG_TYPE} = "deb" ]; then
  # Debian Lenny workaround (--install-layout does not exist)
#   if [ ${DISTRO} = "debian-lenny" ]; then
#     ${PYTHON} setup.py install --root=${BUILD_DIR}
#   else
#     ${PYTHON} setup.py install --install-layout=deb --root=${BUILD_DIR}
#   fi
  cd ../daetools-package
  ${PYTHON} setup.py install --root=${BUILD_DIR}
  cd ${INSTALLATIONS_DIR}

elif [ ${PCKG_TYPE} = "rpm" ]; then
  ${PYTHON} setup.py install --prefix=/usr --root=${BUILD_DIR}

fi

#if [ -d ${BUILD_DIR}/usr/lib ]; then
#  PYTHON_USRLIB=/usr/lib
#else
#  PYTHON_USRLIB=/usr/lib64
#fi
#DAE_TOOLS_DIR=${PYTHON_USRLIB}/python${PYTHON_VERSION}/${SITE_PACK}/${PACKAGE_NAME}

#DAE_TOOLS_DIR=${SITE_PACKAGES_DIR}/${PACKAGE_NAME}

# Delete all .pyc files
find ${BUILD_DIR} -name \*.pyc | xargs rm

# Set execute flag to all python files except __init__.py
#find ${BUILD_DIR} -name \*.py        | xargs chmod +x
#find ${BUILD_DIR} -name \__init__.py | xargs chmod -x

#ICON=${DAE_TOOLS_DIR}/daePlotter/images/app.xpm
#ICON="daetools_main"

#if [ ! ${PLATFORM} = "darwin" ]; then
#   mkdir ${BUILD_DIR}/usr/share
# 
#   # Man page
#   mkdir ${BUILD_DIR}/usr/share/man
#   mkdir ${BUILD_DIR}/usr/share/man/man1
#   gzip -c -9 ../daetools.1 > ${BUILD_DIR}/usr/share/man/man1/daetools.1.gz
# 
#   # Changelog file
#   mkdir ${BUILD_DIR}/usr/share/doc
#   mkdir ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}
#   cp ../copyright ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}
#   gzip -c -9 ../Website/changelog > ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}/changelog.Debian.gz
# 
#   # Shortcuts
#   mkdir ${BUILD_DIR}/usr/share/applications
# 
#   daePlotter_DESKTOP=${BUILD_DIR}/usr/share/applications/daetools-daePlotter.desktop
#   echo "[Desktop Entry]"                                 > ${daePlotter_DESKTOP}
#   echo "Name=daePlotter"                                >> ${daePlotter_DESKTOP}
#   echo "GenericName=Equation-Oriented modelling tool"   >> ${daePlotter_DESKTOP}
#   echo "Comment=DAE Tools Plotter"                      >> ${daePlotter_DESKTOP}
#   echo "Categories=GNOME;Development;"                  >> ${daePlotter_DESKTOP}
#   echo "Exec=/usr/bin/daeplotter"                       >> ${daePlotter_DESKTOP}
#   echo "Icon=${ICON}"                                   >> ${daePlotter_DESKTOP}
#   echo "Terminal=false"                                 >> ${daePlotter_DESKTOP}
#   echo "Type=Application"                               >> ${daePlotter_DESKTOP}
#   echo "StartupNotify=true"                             >> ${daePlotter_DESKTOP}
# 
#   daeExamples_DESKTOP=${BUILD_DIR}/usr/share/applications/daetools-daeExamples.desktop
#   echo "[Desktop Entry]"                                 > ${daeExamples_DESKTOP}
#   echo "Name=DAE Tools Examples"                        >> ${daeExamples_DESKTOP}
#   echo "GenericName=DAE Tools Examples"                 >> ${daeExamples_DESKTOP}
#   echo "Comment=DAE Tools Examples"                     >> ${daeExamples_DESKTOP}
#   echo "Categories=GNOME;Development;"                  >> ${daeExamples_DESKTOP}
#   echo "Exec=/usr/bin/daeexamples"                      >> ${daeExamples_DESKTOP}
#   echo "Icon=${ICON}"                                   >> ${daeExamples_DESKTOP}
#   echo "Terminal=false"                                 >> ${daeExamples_DESKTOP}
#   echo "Type=Application"                               >> ${daeExamples_DESKTOP}
#   echo "StartupNotify=true"                             >> ${daeExamples_DESKTOP}
#fi

if [ ${PCKG_TYPE} = "deb" ]; then
  mkdir ${BUILD_DIR}/DEBIAN

  CONTROL=${BUILD_DIR}/DEBIAN/control
  echo "Package: ${PACKAGE_NAME} "                                                                                                   > ${CONTROL}
  echo "Version: ${VER_MAJOR}.${VER_MINOR}.${VER_BUILD} "                                                                           >> ${CONTROL}
  echo "Architecture: ${ARCH} "                                                                                                     >> ${CONTROL}
  echo "Section: math "                                                                                                             >> ${CONTROL}
  echo "Priority: optional "                                                                                                        >> ${CONTROL}
  echo "Installed-Size: 11,700 "                                                                                                    >> ${CONTROL}
  echo "Maintainer: Dragan Nikolic <dnikolic@daetools.com> "                                                                        >> ${CONTROL}
  echo "Depends: python${PYTHON_VERSION}, libboost-all-dev, python-qt4, python-numpy, python-scipy, python-matplotlib, python-tk"   >> ${CONTROL}
  echo "Description: A cross-platform equation-oriented process modelling, simulation and optimization software."                   >> ${CONTROL}
  echo "Suggests: mayavi2, libumfpack, libamd, libblas3gf, liblapack3gf "                                                           >> ${CONTROL}
  echo "Homepage: http://www.daetools.com "                                                                                         >> ${CONTROL}

  CONFFILES=${BUILD_DIR}/DEBIAN/conffiles
  echo "/etc/daetools/daetools.cfg"   > ${CONFFILES}
  echo "/etc/daetools/bonmin.cfg"    >> ${CONFFILES}

if [ ${BOOST_BUILD_TYPE} = "custom" ]; then 
  SHLIBS=${BUILD_DIR}/DEBIAN/shlibs
  echo "${BOOST_PYTHON_LIB_NAME} ${BOOST_VERSION} ${BOOST_PYTHON_LIB} (>= ${BOOST_MAJOR}:${BOOST_VERSION})"  > ${SHLIBS}
  echo "${BOOST_SYSTEM_LIB_NAME} ${BOOST_VERSION} ${BOOST_SYSTEM_LIB} (>= ${BOOST_MAJOR}:${BOOST_VERSION})" >> ${SHLIBS}
  echo "${BOOST_THREAD_LIB_NAME} ${BOOST_VERSION} ${BOOST_THREAD_LIB} (>= ${BOOST_MAJOR}:${BOOST_VERSION})" >> ${SHLIBS}
fi

#   mkdir ${BUILD_DIR}/usr/share/menu
#   MENU=${BUILD_DIR}/usr/share/menu/${PACKAGE_NAME}
#   echo "?package(${PACKAGE_NAME}):\\"                         > ${MENU}
#   echo "    needs=\"x11\" \\"                                >> ${MENU}
#   echo "    section=\"Applications/Development\" \\"         >> ${MENU}
#   echo "    title=\"daePlotter\" \\"                         >> ${MENU}
#   echo "    icon=\"${ICON}\" \\"                             >> ${MENU}
#   echo "    command=\"/usr/bin/daeplotter\""                 >> ${MENU}
# 
#   EXAMPLES=${BUILD_DIR}/usr/share/menu/${PACKAGE_NAME}-examples
#   echo "?package(${PACKAGE_NAME}-examples):\\"                > ${EXAMPLES}
#   echo "    needs=\"x11\" \\"                                >> ${EXAMPLES}
#   echo "    section=\"Applications/Development\" \\"         >> ${EXAMPLES}
#   echo "    title=\"DAE Tools Examples\" \\"                 >> ${EXAMPLES}
#   echo "    icon=\"${ICON}\" \\"                             >> ${EXAMPLES}
#   echo "    command=\"/usr/bin/daeexamples\""                >> ${EXAMPLES}

  POSTRM=${BUILD_DIR}/DEBIAN/postrm
  echo "#!/bin/sh"                                                                               > ${POSTRM}
  echo "set -e"                                                                                 >> ${POSTRM}
  echo "if [ \"\$1\" = \"configure\" ] && [ -x \"\`which update-menus 2>/dev/null\`\" ]; then " >> ${POSTRM}
  echo "    update-menus"                                                                       >> ${POSTRM}
  echo "fi"                                                                                     >> ${POSTRM}
  echo "case \"\$1\" in "                                                                       >> ${POSTRM}
  echo "    remove) "                                                                           >> ${POSTRM}
  echo "         ldconfig "                                                                     >> ${POSTRM}
  echo "    ;; "                                                                                >> ${POSTRM}
  echo "esac "                                                                                  >> ${POSTRM}
  chmod 0755 ${POSTRM}

  POSTINST=${BUILD_DIR}/DEBIAN/postinst
  echo "#!/bin/sh"                                                                              > ${POSTINST}
  echo "set -e"                                                                                >> ${POSTINST}
  echo "if [ \"\$1\" = \"configure\" ] && [ -x \"\`which update-menus 2>/dev/null\`\" ]; then" >> ${POSTINST}
  echo "    update-menus"                                                                      >> ${POSTINST}
  echo "fi"                                                                                    >> ${POSTINST}
  echo "case \"\$1\" in "                                                                      >> ${POSTINST}
  echo "    configure) "                                                                       >> ${POSTINST}
  echo "        ldconfig "                                                                     >> ${POSTINST}
  echo "    ;; "                                                                               >> ${POSTINST}
  echo "esac "                                                                                 >> ${POSTINST}
  chmod 0755 ${POSTINST}

  fakeroot dpkg -b ${BUILD_DIR}

elif [ ${PCKG_TYPE} = "distutils.tar.gz" ]; then
  tar -czvf ${BUILD_DIR}.tar.gz ${BUILD_DIR}

elif [ ${PCKG_TYPE} = "rpm" ]; then
  cd ${BUILD_DIR}
  tar -czvf ${TGZ} *
  cd ..
  
  SPEC=dae-tools.spec
  echo "%define is_mandrake %(test -e /etc/mandrake-release && echo 1 || echo 0)"   > ${SPEC}
  echo "%define is_suse %(test -e /etc/SuSE-release && echo 1 || echo 0) "         >> ${SPEC}
  echo "%define is_fedora %(test -e /etc/fedora-release && echo 1 || echo 0) "     >> ${SPEC}
  echo "%define dist redhat"        >> ${SPEC}
  echo "%define disttag rh"         >> ${SPEC}
  echo "%if %is_mandrake"           >> ${SPEC}
  echo "%define dist mandrake"      >> ${SPEC}
  echo "%define disttag mdk"        >> ${SPEC}
  echo "%endif"                     >> ${SPEC}
  echo "%if %is_suse"               >> ${SPEC}
  echo "%define dist suse"          >> ${SPEC}
  echo "%define disttag suse"       >> ${SPEC}
  echo "%endif"                     >> ${SPEC}
  echo "%if %is_fedora"             >> ${SPEC}
  echo "%define dist fedora"        >> ${SPEC}
  echo "%define disttag rhfc"       >> ${SPEC}
  echo "%endif"                     >> ${SPEC}

  echo "Summary: DAE Tools: A cross-platform equation-oriented process modelling, simulation and optimization software." >> ${SPEC}
  echo "Name: ${PACKAGE_NAME}"                                                               >> ${SPEC}
  echo "Version: ${VER_MAJOR}.${VER_MINOR}"                                                  >> ${SPEC}
  echo "Release: ${VER_BUILD}"                                                               >> ${SPEC}
  echo "Packager:  Dragan Nikolic dnikolic@daetools.com"                                     >> ${SPEC}
  echo "License: GNU GPL v3"                                                                 >> ${SPEC}
  echo "URL: www.daetools.com"                                                               >> ${SPEC}
  echo "Requires: boost-devel >= 1.41, PyQt4, numpy, scipy, python-matplotlib, blas, lapack" >> ${SPEC}
  echo "ExclusiveArch: ${ARCH_RPM}"                                                          >> ${SPEC}
  echo "Group: Development/Tools"                                                            >> ${SPEC}

  echo "%description"                                                               >> ${SPEC}
  echo "DAE Tools: A cross-platform equation-oriented process modelling, simulation and optimization software." >> ${SPEC}

  echo "%prep"                                                              >> ${SPEC}
  echo "%build"                                                             >> ${SPEC}

  echo "%install"                                                           >> ${SPEC}
  echo "if [ ! -d %{buildroot} ]; then "                                    >> ${SPEC}
  echo "  mkdir %{buildroot} "                                              >> ${SPEC}
  echo "fi "                                                                >> ${SPEC}
  echo "tar -xzf ${INSTALLATIONS_DIR}/${BUILD_DIR}/${TGZ} -C %{buildroot} " >> ${SPEC}

  echo "%files"                                                             >> ${SPEC}
  echo "%defattr(-,root,root,-) "                                           >> ${SPEC}
  echo "/"                                                                  >> ${SPEC}

  echo "%config(noreplace)"                                                 >> ${SPEC}
  echo "/etc/daetools/daetools.cfg"                                         >> ${SPEC}
  echo "/etc/daetools/bonmin.cfg"                                           >> ${SPEC}

  echo "%clean"                                                             >> ${SPEC}
  echo "rm -rf %{buildroot}"                                                >> ${SPEC}

  rpmbuild -bb dae-tools.spec
  cp ~/rpmbuild/RPMS/${ARCH_RPM}/${RPM} ${INSTALLATIONS_DIR}/${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${ARCH}_${DISTRO}.rpm

  rm ${SPEC}

else
  echo "ERROR: undefined type of a package"
  return
fi

exit

# Clean up
if [ -d ${BUILD_DIR} ]; then
  rm -r ${BUILD_DIR}
fi

if [ -d ${PACKAGE_NAME} ]; then
  rm -r ${PACKAGE_NAME}
fi

if [ -d build ]; then
  rm -r build
fi

if [ -e ${SETUP_PY} ]; then
  rm ${SETUP_PY}
fi
