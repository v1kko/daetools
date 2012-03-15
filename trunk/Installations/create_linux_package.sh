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
cd ${INSTALLATIONS_DIR}

VER_MAJOR=
VER_MINOR=
VER_BUILD=
PACKAGE_NAME=daetools
PCKG_TYPE=
ARCH=
LIB=
ARCH_RPM=
SITE_PACK=
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
  SO_EXT="so"
  DISTRIBUTOR_ID=`echo $(lsb_release -si) | tr "[:upper:]" "[:lower:]"`
  CODENAME=`echo $(lsb_release -sc) | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}-${CODENAME}

elif [ ${PLATFORM} = "gnu/kfreebsd" ]; then
  SO_EXT="so"
  ARCH=kfreebsd-${ARCH}
  DISTRIBUTOR_ID=`echo $(lsb_release -si) | tr "[:upper:]" "[:lower:]"`
  CODENAME=`echo $(lsb_release -sc) | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}-${CODENAME}

elif [ ${PLATFORM} = "darwin" ]; then
  SO_EXT="so"
  DISTRIBUTOR_ID="macosx"
  CODENAME=`echo $(sw_vers -productVersion) | tr "[:upper:]" "[:lower:]"`
  DISTRO=${DISTRIBUTOR_ID}-${CODENAME}

else
  echo "ERROR: undefined platform: ${PLATFORM}"
  exit
fi
 
PYTHON_VERSION=`python -c "import sys; print (\"%d.%d\" % (sys.version_info[0], sys.version_info[1]))"`
VER_MAJOR=`python -c "import imp; pyCore = imp.load_dynamic('pyCore', '${RELEASE_DIR}/pyCore.${SO_EXT}'); print (\"%d\" % pyCore.daeVersionMajor())"`
VER_MINOR=`python -c "import imp; pyCore = imp.load_dynamic('pyCore', '${RELEASE_DIR}/pyCore.${SO_EXT}'); print (\"%d\" % pyCore.daeVersionMinor())"`
VER_BUILD=`python -c "import imp; pyCore = imp.load_dynamic('pyCore', '${RELEASE_DIR}/pyCore.${SO_EXT}'); print (\"%d\" % pyCore.daeVersionBuild())"`
VERSION=${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}
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
  if [ ${CODENAME} = "lenny" ]; then
    SITE_PACK="site-packages"
  else
    SITE_PACK="dist-packages"
  fi

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  PCKG_TYPE="deb"
  SITE_PACK="dist-packages"

elif [ ${DISTRIBUTOR_ID} = "linuxmint" ]; then
  PCKG_TYPE="deb"
  SITE_PACK="dist-packages"

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  PCKG_TYPE="rpm"
  SITE_PACK="site-packages"

elif [ ${DISTRIBUTOR_ID} = "centos" ]; then
  PCKG_TYPE="rpm"
  SITE_PACK="site-packages"

elif [ ${DISTRIBUTOR_ID} = "macosx" ]; then
  PCKG_TYPE="dmg"
  SITE_PACK="site-packages"

else
  echo "ERROR: undefined type of a package"
  exit
fi

TGZ=${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${ARCH}.tar.gz
RPM=${PACKAGE_NAME}-${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}.${ARCH_RPM}.rpm
BUILD_DIR=${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}_${ARCH}_${DISTRO}
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
echo "    site-packages dir: " ${SITE_PACK}
echo "    /usr/lib dir:      " ${USRLIB}
echo " " 
read -p " Proceed [y/n]? " do_proceed
case ${do_proceed} in
  [Nn]* ) echo "Aborting ..."
          exit;;
      * ) break;;
esac

if [ ${PCKG_TYPE} = "tgz" ]; then
  INSTALL_DIR=${PACKAGE_NAME}-${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}
  mkdir ${INSTALL_DIR}
  mkdir ${INSTALL_DIR}/lib
  mkdir ${INSTALL_DIR}/include
  mkdir ${INSTALL_DIR}/include/Core
  mkdir ${INSTALL_DIR}/include/Activity
  mkdir ${INSTALL_DIR}/include/DataReporting
  mkdir ${INSTALL_DIR}/include/IDAS_DAESolver
  mkdir ${INSTALL_DIR}/include/BONMIN_MINLPSolver
  mkdir ${INSTALL_DIR}/include/LA_SuperLU
  mkdir ${INSTALL_DIR}/TestPrograms

  cp ../compile_libraries_linux.sh            ${INSTALL_DIR}

  cp ../release/libcdaeActivity.a             ${INSTALL_DIR}/lib
  cp ../release/libcdaeBONMIN_MINLPSolver.a   ${INSTALL_DIR}/lib
  cp ../release/libcdaeCore.a                 ${INSTALL_DIR}/lib
  cp ../release/libcdaeDataReporting.a        ${INSTALL_DIR}/lib
  cp ../release/libcdaeIDAS_DAESolver.a       ${INSTALL_DIR}/lib
  cp ../release/libcdaeIPOPT_NLPSolver.a      ${INSTALL_DIR}/lib
  cp ../release/libcdaeNLOPT_NLPSolver.a      ${INSTALL_DIR}/lib
  cp ../release/libcdaeSuperLU_LASolver.a     ${INSTALL_DIR}/lib
  cp ../release/libcdaeSuperLU_MT_LASolver.a  ${INSTALL_DIR}/lib

  cp ../dae.h                                           ${INSTALL_DIR}/include
  cp ../dae_develop.h                                   ${INSTALL_DIR}/include
  cp ../config.h                                        ${INSTALL_DIR}/include
  cp ../Core/definitions.h                              ${INSTALL_DIR}/include/Core
  cp ../Core/xmlfile.h                                  ${INSTALL_DIR}/include/Core
  cp ../Core/coreimpl.h                                 ${INSTALL_DIR}/include/Core
  cp ../Core/helpers.h                                  ${INSTALL_DIR}/include/Core
  cp ../Core/base_logging.h                             ${INSTALL_DIR}/include/Core
  cp ../Core/class_factory.h                            ${INSTALL_DIR}/include/Core
  cp ../Activity/base_activities.h                      ${INSTALL_DIR}/include/Activity
  cp ../Activity/simulation.h                           ${INSTALL_DIR}/include/Activity
  cp ../DataReporting/datareporters.h                   ${INSTALL_DIR}/include/DataReporting
  cp ../DataReporting/base_data_reporters_receivers.h   ${INSTALL_DIR}/include/DataReporting
  cp ../IDAS_DAESolver/base_solvers.h                   ${INSTALL_DIR}/include/IDAS_DAESolver
  cp ../IDAS_DAESolver/ida_solver.h                     ${INSTALL_DIR}/include/IDAS_DAESolver
  cp ../BONMIN_MINLPSolver/base_solvers.h               ${INSTALL_DIR}/include/BONMIN_MINLPSolver
  cp ../LA_SuperLU/superlu_solvers.h                    ${INSTALL_DIR}/include/LA_SuperLU
  
  cp ../dae.pri                                         ${INSTALL_DIR}
  cp ../TestPrograms/TestPrograms.pro                   ${INSTALL_DIR}/TestPrograms
  cp ../TestPrograms/main.cpp                           ${INSTALL_DIR}/TestPrograms
  cp ../TestPrograms/*tutorial*.h                       ${INSTALL_DIR}/TestPrograms
  cp ../TestPrograms/whats_the_time.h                   ${INSTALL_DIR}/TestPrograms
  cp ../TestPrograms/variable_types.h                   ${INSTALL_DIR}/TestPrograms

  tar -czvf ${TGZ} ${INSTALL_DIR}
  rm -r ${INSTALL_DIR}
  exit
fi

if [ -d ${PACKAGE_NAME} ]; then
  rm -r ${PACKAGE_NAME}
fi
mkdir ${PACKAGE_NAME}
mkdir ${PACKAGE_NAME}/docs
mkdir ${PACKAGE_NAME}/docs/images
mkdir ${PACKAGE_NAME}/docs/api_ref
mkdir ${PACKAGE_NAME}/daePlotter
mkdir ${PACKAGE_NAME}/daePlotter/images
mkdir ${PACKAGE_NAME}/examples
mkdir ${PACKAGE_NAME}/examples/images
mkdir ${PACKAGE_NAME}/pyDAE
mkdir ${PACKAGE_NAME}/daeSimulator
mkdir ${PACKAGE_NAME}/daeSimulator/images
mkdir ${PACKAGE_NAME}/solvers
mkdir ${PACKAGE_NAME}/model_library

if [ -d ${BUILD_DIR} ]; then
  rm -r ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
mkdir ${BUILD_DIR}/usr
mkdir ${BUILD_DIR}/usr/bin
mkdir ${BUILD_DIR}${USRLIB}

# Python extension modules and LA solvers
cp ../release/pyCore.${SO_EXT}             ${PACKAGE_NAME}/pyDAE
cp ../release/pyActivity.${SO_EXT}         ${PACKAGE_NAME}/pyDAE
cp ../release/pyDataReporting.${SO_EXT}    ${PACKAGE_NAME}/pyDAE
cp ../release/pyIDAS.${SO_EXT}             ${PACKAGE_NAME}/pyDAE
cp ../release/pyUnits.${SO_EXT}            ${PACKAGE_NAME}/pyDAE

if [ -e ../release/pyBONMIN.${SO_EXT} ]; then
  cp ../release/pyBONMIN.${SO_EXT}          ${PACKAGE_NAME}/solvers
fi

if [ -e ../release/pyIPOPT.${SO_EXT} ]; then
  cp ../release/pyIPOPT.${SO_EXT}          ${PACKAGE_NAME}/solvers
fi

if [ -e ../release/pyNLOPT.${SO_EXT} ]; then
  cp ../release/pyNLOPT.${SO_EXT}          ${PACKAGE_NAME}/solvers
fi

#if [ -e ../release/pyAmdACML.${SO_EXT} ]; then
#  cp ../release/pyAmdACML.${SO_EXT}          ${PACKAGE_NAME}/solvers
#fi

#if [ -e ../release/pyIntelMKL.${SO_EXT} ]; then
#  cp ../release/pyIntelMKL.${SO_EXT}         ${PACKAGE_NAME}/solvers
#fi

#if [ -e ../release/pyLapack.${SO_EXT} ]; then
#  cp ../release/pyLapack.${SO_EXT}           ${PACKAGE_NAME}/solvers
#fi

#if [ -e ../release/pyMagma.${SO_EXT} ]; then
#  cp ../release/pyMagma.${SO_EXT}             ${PACKAGE_NAME}/solvers
#fi

#if [ -e ../release/pyCUSP.${SO_EXT} ]; then
#  cp ../release/pyCUSP.${SO_EXT}              ${PACKAGE_NAME}/solvers
#fi

if [ -e ../release/pySuperLU.${SO_EXT} ]; then
  cp ../release/pySuperLU.${SO_EXT}           ${PACKAGE_NAME}/solvers
fi
if [ -e ../release/pySuperLU_MT.${SO_EXT} ]; then
  cp ../release/pySuperLU_MT.${SO_EXT}        ${PACKAGE_NAME}/solvers
fi
if [ -e ../release/pySuperLU_CUDA.${SO_EXT} ]; then
  cp ../release/pySuperLU_CUDA.${SO_EXT}      ${PACKAGE_NAME}/solvers
fi

#if [ -e ../release/pyIntelPardiso.${SO_EXT} ]; then
#  cp ../release/pyIntelPardiso.${SO_EXT}     ${PACKAGE_NAME}/solvers
#fi

if [ -e ../release/pyTrilinos.${SO_EXT} ]; then
  cp ../release/pyTrilinos.${SO_EXT}   ${PACKAGE_NAME}/solvers
fi

# Licences
cp ../licence*                                   ${PACKAGE_NAME}
cp ../ReadMe.txt                                 ${PACKAGE_NAME}

# Python files
cp ../python-files/daetools__init__.py           ${PACKAGE_NAME}/__init__.py
cp ../python-files/daeLogs.py                    ${PACKAGE_NAME}/pyDAE
cp ../python-files/WebView_ui.py                 ${PACKAGE_NAME}/pyDAE
cp ../python-files/WebViewDialog.py              ${PACKAGE_NAME}/pyDAE
cp ../python-files/daeLogs.py                    ${PACKAGE_NAME}/pyDAE
cp ../python-files/daeVariableTypes.py           ${PACKAGE_NAME}/pyDAE
cp ../python-files/daeDataReporters.py           ${PACKAGE_NAME}/pyDAE
cp ../python-files/pyDAE__init__.py              ${PACKAGE_NAME}/pyDAE/__init__.py
cp ../python-files/solvers__init__.py            ${PACKAGE_NAME}/solvers/__init__.py
cp ../python-files/aztecoo_options.py            ${PACKAGE_NAME}/solvers
cp ../python-files/daeMinpackLeastSq.py          ${PACKAGE_NAME}/solvers
cp ../python-files/model_library__init__.py      ${PACKAGE_NAME}/model_library/__init__.py

# daeSimulator
cp ../python-files/daeSimulator/__init__.py      ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/daeSimulator.py  ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/Simulator_ui.py  ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/images/*.*       ${PACKAGE_NAME}/daeSimulator/images

# daePlotter
cp ../python-files/daePlotter/*.py               ${PACKAGE_NAME}/daePlotter
cp ../python-files/daePlotter/images/*.*         ${PACKAGE_NAME}/daePlotter/images

# Model Library
cp ../python-files/model_library/*.py            ${PACKAGE_NAME}/model_library

# Examples and Tutorials
cp ../python-files/examples/*.css                ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.xsl                ${PACKAGE_NAME}/examples
cp ../python-files/examples/*tutorial*.*         ${PACKAGE_NAME}/examples
cp ../python-files/examples/*RunExamples*.py     ${PACKAGE_NAME}/examples
cp ../python-files/examples/*whats_the_time*.*   ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.init               ${PACKAGE_NAME}/examples
cp ../python-files/examples/images/*.*           ${PACKAGE_NAME}/examples/images

# Documentation
cp ../python-files/api_ref/*.html  ${PACKAGE_NAME}/docs/api_ref

# Strip python extension modules
#find ${PACKAGE_NAME}/pyDAE   -name \*.${SO_EXT}* | xargs strip
#find ${PACKAGE_NAME}/solvers -name \*.${SO_EXT}* | xargs strip

echo "#!/usr/bin/env python " > setup.py
echo "import sys " >> setup.py
echo "from distutils.core import setup " >> setup.py
echo " " >> setup.py
echo "setup(name='${PACKAGE_NAME}', " >> setup.py
echo "      version='${VERSION}', " >> setup.py
echo "      description='DAE Tools', " >> setup.py
echo "      long_description='A cross-platform equation-oriented process modelling software (pyDAE modules).', " >> setup.py
echo "      author='Dragan Nikolic', " >> setup.py
echo "      author_email='dnikolic@daetools.com', " >> setup.py
echo "      url='http://www.daetools.com', " >> setup.py
echo "      license='GNU GPL v3', " >> setup.py
echo "      platforms='${ARCH}', " >> setup.py
echo "      packages=['${PACKAGE_NAME}'], " >> setup.py
echo "      package_dir={'${PACKAGE_NAME}': '${PACKAGE_NAME}'}, " >> setup.py
echo "      package_data={'${PACKAGE_NAME}': ['*.*', 'pyDAE/*.*', 'model_library/*.*', 'examples/*.*', 'examples/images/*.*', 'docs/*.*', 'docs/images/*.*', 'docs/api_ref/*.*', 'daeSimulator/*.*', 'daeSimulator/images/*.*', 'daePlotter/*.*', 'daePlotter/images/*.*', 'solvers/*.*']} " >> setup.py
echo "      ) " >> setup.py
echo " " >> setup.py

if [ ${PCKG_TYPE} = "deb" ]; then
  # Debian Lenny workaround (--install-layout does not exist)
  if [ ${DISTRO} = "debian-lenny" ]; then
    python setup.py install --root=${BUILD_DIR}
  else
    python setup.py install --install-layout=deb --root=${BUILD_DIR}
  fi

elif [ ${PCKG_TYPE} = "dmg" ]; then
  python setup.py install --root=${BUILD_DIR}

elif [ ${PCKG_TYPE} = "rpm" ]; then
  python setup.py install --prefix=/usr --root=${BUILD_DIR}
fi

if [ -d ${BUILD_DIR}/usr/lib ]; then
  PYTHON_USRLIB=/usr/lib
else
  PYTHON_USRLIB=/usr/lib64
fi
DAE_TOOLS_DIR=${PYTHON_USRLIB}/python${PYTHON_VERSION}/${SITE_PACK}/${PACKAGE_NAME}

# Delete all .pyc files
find ${BUILD_DIR} -name \*.pyc | xargs rm

# Set execute flag to all python files except __init__.py
#find ${BUILD_DIR} -name \*.py        | xargs chmod +x
#find ${BUILD_DIR} -name \__init__.py | xargs chmod -x

# Change permissions and strip libraries in /usr/lib(64) 
#chmod -x ${BUILD_DIR}${USRLIB}/*.${SO_EXT}*
#find ${BUILD_DIR}${USRLIB} -name \*.${SO_EXT}* | xargs strip

ICON=${DAE_TOOLS_DIR}/daePlotter/images/app.xpm

echo "#!/bin/sh"                       > ${USRBIN}/daeplotter
echo "cd ${DAE_TOOLS_DIR}/daePlotter" >> ${USRBIN}/daeplotter
echo "python daePlotter.py"           >> ${USRBIN}/daeplotter
chmod +x ${USRBIN}/daeplotter

echo "#!/bin/sh"                     > ${USRBIN}/daeexamples
echo "cd ${DAE_TOOLS_DIR}/examples" >> ${USRBIN}/daeexamples
echo "python daeRunExamples.py"     >> ${USRBIN}/daeexamples
chmod +x ${USRBIN}/daeexamples

mkdir ${BUILD_DIR}/usr/share

# Man page
mkdir ${BUILD_DIR}/usr/share/man
mkdir ${BUILD_DIR}/usr/share/man/man1
gzip -c -9 ../daetools.1 > ${BUILD_DIR}/usr/share/man/man1/daetools.1.gz

# Changelog file
mkdir ${BUILD_DIR}/usr/share/doc
mkdir ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}
cp ../copyright ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}
gzip -c -9 ../Website/changelog > ${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}/changelog.Debian.gz

# Config
mkdir ${BUILD_DIR}/etc
mkdir ${BUILD_DIR}/etc/daetools

cp ../daetools.cfg  ${BUILD_DIR}/etc/daetools
cp ../bonmin.cfg    ${BUILD_DIR}/etc/daetools
chmod go-wx ${BUILD_DIR}/etc/daetools/daetools.cfg
chmod go-wx ${BUILD_DIR}/etc/daetools/bonmin.cfg

# Shortcuts
mkdir ${BUILD_DIR}/usr/share/applications

daePlotter_DESKTOP=${BUILD_DIR}/usr/share/applications/daetools-daePlotter.desktop
echo "[Desktop Entry]"                                 > ${daePlotter_DESKTOP}
echo "Name=daePlotter"                                >> ${daePlotter_DESKTOP}
echo "GenericName=Equation-Oriented modelling tool"   >> ${daePlotter_DESKTOP}
echo "Comment=DAE Tools Plotter"                      >> ${daePlotter_DESKTOP}
echo "Categories=GNOME;Development;"                  >> ${daePlotter_DESKTOP}
echo "Exec=/usr/bin/daeplotter"                       >> ${daePlotter_DESKTOP}
echo "Icon=${ICON}"                                   >> ${daePlotter_DESKTOP}
echo "Terminal=false"                                 >> ${daePlotter_DESKTOP}
echo "Type=Application"                               >> ${daePlotter_DESKTOP}
echo "StartupNotify=true"                             >> ${daePlotter_DESKTOP}

daeExamples_DESKTOP=${BUILD_DIR}/usr/share/applications/daetools-Examples.desktop
echo "[Desktop Entry]"                                 > ${daeExamples_DESKTOP}
echo "Name=DAE Tools Examples"                        >> ${daeExamples_DESKTOP}
echo "GenericName=DAE Tools Examples"                 >> ${daeExamples_DESKTOP}
echo "Comment=DAE Tools Examples"                     >> ${daeExamples_DESKTOP}
echo "Categories=GNOME;Development;"                  >> ${daeExamples_DESKTOP}
echo "Exec=/usr/bin/daeexamples"                      >> ${daeExamples_DESKTOP}
echo "Icon=${ICON}"                                   >> ${daeExamples_DESKTOP}
echo "Terminal=false"                                 >> ${daeExamples_DESKTOP}
echo "Type=Application"                               >> ${daeExamples_DESKTOP}
echo "StartupNotify=true"                             >> ${daeExamples_DESKTOP}

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

  #SHLIBS=${BUILD_DIR}/DEBIAN/shlibs
  #echo "libsuperlu 4.1 libsuperlu.so.4.1 (>= 4:4.1)"          > ${SHLIBS}
  #echo "libsuperlu_mt 2.0 libsuperlu_mt.so.2.0 (>= 2:2.0)"   >> ${SHLIBS}

  mkdir ${BUILD_DIR}/usr/share/menu
  MENU=${BUILD_DIR}/usr/share/menu/${PACKAGE_NAME}
  echo "?package(${PACKAGE_NAME}):\\"                         > ${MENU}
  echo "    needs=\"x11\" \\"                                >> ${MENU}
  echo "    section=\"Applications/Development\" \\"         >> ${MENU}
  echo "    title=\"daePlotter\" \\"                         >> ${MENU}
  echo "    icon=\"${ICON}\" \\"                             >> ${MENU}
  echo "    command=\"/usr/bin/daeplotter\""                 >> ${MENU}

  EXAMPLES=${BUILD_DIR}/usr/share/menu/${PACKAGE_NAME}-examples
  echo "?package(${PACKAGE_NAME}-examples):\\"                > ${MENU}
  echo "    needs=\"x11\" \\"                                >> ${MENU}
  echo "    section=\"Applications/Development\" \\"         >> ${MENU}
  echo "    title=\"DAE Tools Examples\" \\"                 >> ${MENU}
  echo "    icon=\"${ICON}\" \\"                             >> ${MENU}
  echo "    command=\"/usr/bin/daeexamples\""                >> ${MENU}

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
  echo "chmod -R o+w ${DAE_TOOLS_DIR}/examples"                                                >> ${POSTINST}
  chmod 0755 ${POSTINST}

  fakeroot dpkg -b ${BUILD_DIR}

elif [ ${PCKG_TYPE} = "dmg" ]; then
  MAC_INSTALL_SCRIPT=${BUILD_DIR}/install
  echo "#!/bin/sh"                                    > ${MAC_INSTALL_SCRIPT}
  echo "set -e"                                      >> ${MAC_INSTALL_SCRIPT}
  echo "echo Copying daetools installation filesÉ"   >> ${MAC_INSTALL_SCRIPT}
  echo "sudo cp -vR /Volumes/${BUILD_DIR}/*  /"      >> ${MAC_INSTALL_SCRIPT}
  chmod 0755 ${MAC_INSTALL_SCRIPT}
  
  hdiutil create ${BUILD_DIR}.dmg -srcfolder ./${BUILD_DIR} -ov

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
  echo "Name: ${PACKAGE_NAME}"                                                              >> ${SPEC}
  echo "Version: ${VER_MAJOR}.${VER_MINOR}"                                                 >> ${SPEC}
  echo "Release: ${VER_BUILD}"                                                              >> ${SPEC}
  echo "Packager:  Dragan Nikolic dnikolic@daetools.com"                                    >> ${SPEC}
  echo "License: GNU GPL v3"                                                                >> ${SPEC}
  echo "URL: www.daetools.com"                                                              >> ${SPEC}
  echo "Requires: boost-devel >= 1.41, PyQt4, numpy, scipy, python-matplotlib, blas, lapack">> ${SPEC}
  echo "ExclusiveArch: ${ARCH_RPM}"                                                 >> ${SPEC}
  echo "Group: Development/Tools"                                                   >> ${SPEC}

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

# Clean up
rm -r ${PACKAGE_NAME}
rm -r ${BUILD_DIR}
rm -r build
rm setup.py
