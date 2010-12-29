#!/bin/sh
#                            $1        $2        $3         $4             $5                                                         $6
# sh create_linux_package.sh ver_major ver_minor ver_build  pckg_type      Distro                          OR  boost version          Python version
# sh create_linux_package.sh 1         0         0          deb/tgz/rpm    Debian-6/Ubuntu-10.4/Fedora-13      1.40/1.41/1.42 ...     2.6
#
# Examples:
# sh create_linux_package.sh 1 0 2 deb Debian-6
# sh create_linux_package.sh 1 0 2 tgz 1.41

set -e

if [ "$1" = "-help" ]; then
  echo "Usage:"
  echo "                        1     2     3      4              5                            6  "
  echo "create_linux_package.sh major minor build  pckg_type      Distro    OR  boost version  Python version"
  echo "create_linux_package.sh 1     0     0      deb/tgz/rpm    Debian-6      1.40           2.6"
  echo " "
  echo "If pckg = tgz then boost version should be supplied"
  echo "Distros: Debian-5, Debian-6, Ubuntu-10.4, Ubuntu-10.10, Fedora-13"
  echo " "
  echo "Examples:"
  echo "sh create_linux_package.sh 1 0 2 deb Debian-6 2.6"
  echo "sh create_linux_package.sh 1 0 2 tgz 1.41 2.6"
  return
fi

VER_MAJOR=$1
VER_MINOR=$2
VER_BUILD=$3
PACKAGE_NAME=daetools
PCKG_TYPE=
PLATFORM=
VERSION=${VER_MAJOR}.${VER_MINOR}.${VER_BUILD}
HOST_ARCH=`uname -m`
ARCH=
LIB=
ARCH_RPM=
SITE_PACK=
INSTALLATIONS_DIR=`pwd`
DISTRO=$5
BOOST=boost_$5
PYTHON_VERSION=$6
IDAS=../idas-1.0.0/build
BONMIN=../bonmin/build

if [ ${HOST_ARCH} = "x86_64" ]; then
  LIB=lib64
  ARCH=amd64
  ARCH_RPM=x86_64
fi 
if [ ${HOST_ARCH} = "armv5tejl" ]; then
  LIB=lib
  ARCH=armel
  ARCH_RPM=armel
fi
if [ ${HOST_ARCH} = "i386" ]; then
  LIB=lib
  ARCH=i386
  ARCH_RPM=i386
fi
if [ ${HOST_ARCH} = "i686" ]; then
  LIB=lib
  ARCH=i386
  ARCH_RPM=i386
fi

if [ $4 = "deb" ]; then
  PCKG_TYPE=deb
  PLATFORM=linux
  if [ ${DISTRO} = "Debian-5" ]; then
    SITE_PACK=site-packages
  else
    SITE_PACK=dist-packages
  fi
elif [ $4 = "tgz" ]; then  
  PCKG_TYPE=tgz
  PLATFORM=linux
  SITE_PACK=site-packages
elif [ $4 = "rpm" ]; then  
  PCKG_TYPE=rpm
  PLATFORM=linux
  SITE_PACK=site-packages
else  
  echo "ERROR: undefined type of a package"  
  return
fi

TRILINOS=../trilinos-10.4.0-Source/build-extras/lib

echo " "
echo "****************************************************************"
echo "*            SETTINGS                                          *"
echo "****************************************************************"
echo "  Version:           " ${VERSION}
echo "  Platform:          " ${PLATFORM}
echo "  Package type:      " ${PCKG_TYPE}
echo "  Host-Architecture: " ${HOST_ARCH}
echo "  Architecture:      " ${ARCH}
echo "  Lib prefix:        " ${LIB}
echo "  Install. dir:      " ${INSTALLATIONS_DIR}
echo "  Python pckg. dir:  " ${SITE_PACK}
echo "****************************************************************"
echo " "

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

# Python extension modules and LA solvers
cp ../release/pyCore.so             ${PACKAGE_NAME}/pyDAE
cp ../release/pyActivity.so         ${PACKAGE_NAME}/pyDAE
cp ../release/pyDataReporting.so    ${PACKAGE_NAME}/pyDAE
cp ../release/pyIDAS.so             ${PACKAGE_NAME}/pyDAE
cp ../release/pyBONMIN.so           ${PACKAGE_NAME}/pyDAE

mkdir ${PACKAGE_NAME}/pyAmdACML
if [ -e ../release/pyAmdACML.so ]; then
  cp ../release/pyAmdACML.so          ${PACKAGE_NAME}/pyAmdACML
fi

mkdir ${PACKAGE_NAME}/pyIntelMKL
if [ -e ../release/pyIntelMKL.so ]; then
  cp ../release/pyIntelMKL.so         ${PACKAGE_NAME}/pyIntelMKL
fi

mkdir ${PACKAGE_NAME}/pyLapack
if [ -e ../release/pyLapack.so ]; then
  cp ../release/pyLapack.so           ${PACKAGE_NAME}/pyLapack
fi

mkdir ${PACKAGE_NAME}/pyIntelPardiso
if [ -e ../release/pyIntelPardiso.so ]; then
  cp ../release/pyIntelPardiso.so     ${PACKAGE_NAME}/pyIntelPardiso
fi

mkdir ${PACKAGE_NAME}/pyTrilinosAmesos
if [ -e ../release/pyTrilinosAmesos.so ]; then
  cp ../release/pyTrilinosAmesos.so   ${PACKAGE_NAME}/pyTrilinosAmesos
fi

# Licences
cp ../licence*                                   ${PACKAGE_NAME}
cp ../ReadMe.txt                                 ${PACKAGE_NAME}

# Python files
cp ../python-files/daeLogs.py                    ${PACKAGE_NAME}/pyDAE
cp ../python-files/daetools__init__.py           ${PACKAGE_NAME}/__init__.py
cp ../python-files/pyDAE__init__.py              ${PACKAGE_NAME}/pyDAE/__init__.py
cp ../python-files/pyAmdACML__init__.py          ${PACKAGE_NAME}/pyAmdACML/__init__.py
cp ../python-files/pyIntelMKL__init__.py         ${PACKAGE_NAME}/pyIntelMKL/__init__.py
cp ../python-files/pyLapack__init__.py           ${PACKAGE_NAME}/pyLapack/__init__.py
cp ../python-files/pyIntelPardiso__init__.py     ${PACKAGE_NAME}/pyIntelPardiso/__init__.py
cp ../python-files/pyTrilinosAmesos__init__.py   ${PACKAGE_NAME}/pyTrilinosAmesos/__init__.py
cp ../python-files/WebView_ui.py                 ${PACKAGE_NAME}/pyDAE
cp ../python-files/WebViewDialog.py              ${PACKAGE_NAME}/pyDAE

# daeSimulator
cp ../python-files/daeSimulator/__init__.py      ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/daeSimulator.py  ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/Simulator_ui.py  ${PACKAGE_NAME}/daeSimulator
cp ../python-files/daeSimulator/images/*.*       ${PACKAGE_NAME}/daeSimulator/images

# daePlotter
cp ../python-files/daePlotter/*.py          ${PACKAGE_NAME}/daePlotter
cp ../python-files/daePlotter/images/*.*    ${PACKAGE_NAME}/daePlotter/images

# Examples and Tutorials
cp ../python-files/examples/*.css        ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.html       ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.xsl        ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.xml        ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.py         ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.out        ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.png        ${PACKAGE_NAME}/examples
cp ../python-files/examples/*.init       ${PACKAGE_NAME}/examples
cp ../python-files/examples/images/*.*   ${PACKAGE_NAME}/examples/images

# Website
cp ../Website/images/*.png    ${PACKAGE_NAME}/docs/images                    
cp ../Website/images/*.css    ${PACKAGE_NAME}/docs/images                    
cp ../Website/images/*.gif    ${PACKAGE_NAME}/docs/images                    
cp ../Website/images/*.jpg    ${PACKAGE_NAME}/docs/images                    
cp ../Website/*.html          ${PACKAGE_NAME}/docs                
cp ../Website/api_ref/*.html  ${PACKAGE_NAME}/docs/api_ref           
rm ${PACKAGE_NAME}/docs/downloads.html

echo "#!/usr/bin/env python " > setup.py
echo "import sys " >> setup.py
echo "from distutils.core import setup " >> setup.py
echo " " >> setup.py
echo "setup(name='${PACKAGE_NAME}', " >> setup.py
echo "      version='${VERSION}', " >> setup.py
echo "      description='DAE Tools', " >> setup.py
echo "      long_description='A cross-platform equation-oriented process modelling software (pyDAE and cDAE modules).', " >> setup.py
echo "      author='Dragan Nikolic', " >> setup.py
echo "      author_email='dnikolic@daetools.com', " >> setup.py
echo "      url='http://www.daetools.com', " >> setup.py
echo "      license='GNU GPL v3', " >> setup.py
echo "      platforms='${ARCH}', " >> setup.py
echo "      packages=['${PACKAGE_NAME}'], " >> setup.py
echo "      package_dir={'${PACKAGE_NAME}': '${PACKAGE_NAME}'}, " >> setup.py
echo "      package_data={'${PACKAGE_NAME}': ['*.*', 'pyDAE/*.*', 'examples/*.*', 'examples/images/*.*', 'docs/*.*', 'docs/images/*.*', 'docs/api_ref/*.*', 'daeSimulator/*.*', 'daeSimulator/images/*.*', 'daePlotter/*.*', 'daePlotter/images/*.*', 'cDAE/include/*.*', 'cDAE/include/Core/*.*', 'cDAE/include/DataReporters/*.*', 'cDAE/include/Simulation/*.*', 'cDAE/include/Solver/*.*', 'cDAE/lib/*.*', 'pyAmdACML/*.*', 'pyIntelMKL/*.*', 'pyLapack/*.*', 'pyIntelPardiso/*.*', 'pyAtlas/*.*', 'pyTrilinosAmesos/*.*']} " >> setup.py
echo "      ) " >> setup.py
echo " " >> setup.py

TGZ=${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}_${ARCH}.tar.gz
RPM=${PACKAGE_NAME}-${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}.${ARCH_RPM}.rpm
BUILD_DIR=${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}_${ARCH}_${DISTRO}

if [ -d ${BUILD_DIR} ]; then
  rm -r ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}

if [ ${PCKG_TYPE} = "tgz" ]; then
  python setup.py install --prefix=/usr --root=${BUILD_DIR} 
elif [ ${PCKG_TYPE} = "deb" ]; then
  # Debian Lenny workaround (--install-layout does not exist)
  if [ ${DISTRO} = "Debian-5" ]; then
    python setup.py install --root=${BUILD_DIR}
  else
    python setup.py install --install-layout=deb --root=${BUILD_DIR}
  fi
elif [ ${PCKG_TYPE} = "rpm" ]; then
  python setup.py install --prefix=/usr --root=${BUILD_DIR} 
fi

# Delete all .pyc files
find ${BUILD_DIR} -name \*.pyc | xargs rm

# Set execute flag to all python files except __init__.py
#find ${BUILD_DIR} -name \*.py        | xargs chmod +x 
#find ${BUILD_DIR} -name \__init__.py | xargs chmod -x 

# If python is installed in /usr/lib then copy files there; otherwise to /usr/lib64
if [ -d ${BUILD_DIR}/usr/lib ]; then
  USRLIB=/usr/lib
else
  USRLIB=/usr/lib64
fi
# Check if USRLIB exists; if not make it
if [ ! -d ${BUILD_DIR}${USRLIB} ]; then
  mkdir ${BUILD_DIR}${USRLIB}
fi
# Check if lib64 exists; if not make it
if [ ! -d ${BUILD_DIR}/usr/${LIB} ]; then
  mkdir ${BUILD_DIR}/usr/${LIB}
fi
mkdir ${BUILD_DIR}/usr/bin

USRBIN=${BUILD_DIR}/usr/bin
DAE_TOOLS_DIR=${USRLIB}/python${PYTHON_VERSION}/${SITE_PACK}/${PACKAGE_NAME}
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

# cDAE Files
mkdir ${BUILD_DIR}/usr/include
mkdir ${BUILD_DIR}/usr/include/daetools
mkdir ${BUILD_DIR}/usr/include/daetools/Core
mkdir ${BUILD_DIR}/usr/include/daetools/Activity
mkdir ${BUILD_DIR}/usr/include/daetools/DataReporting
mkdir ${BUILD_DIR}/usr/include/daetools/IDAS_DAESolver
mkdir ${BUILD_DIR}/usr/include/daetools/BONMIN_MINLPSolver

mkdir ${BUILD_DIR}/usr/${LIB}/daetools

# Include files
cp ../config.h       ${BUILD_DIR}/usr/include/daetools
cp ../dae.h          ${BUILD_DIR}/usr/include/daetools
cp ../dae_develop.h  ${BUILD_DIR}/usr/include/daetools

cp ../Core/definitions.h    ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/xmlfile.h        ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/helpers.h        ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/base_logging.h   ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/macros.h         ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/class_factory.h  ${BUILD_DIR}/usr/include/daetools/Core
cp ../Core/coreimpl.h       ${BUILD_DIR}/usr/include/daetools/Core

cp ../Activity/base_activities.h  ${BUILD_DIR}/usr/include/daetools/Activity
cp ../Activity/simulation.h       ${BUILD_DIR}/usr/include/daetools/Activity

cp ../DataReporting/datareporters.h                 ${BUILD_DIR}/usr/include/daetools/DataReporting
cp ../DataReporting/base_data_reporters_receivers.h ${BUILD_DIR}/usr/include/daetools/DataReporting

cp ../IDAS_DAESolver/base_solvers.h  ${BUILD_DIR}/usr/include/daetools/IDAS_DAESolver
cp ../IDAS_DAESolver/ida_solver.h    ${BUILD_DIR}/usr/include/daetools/IDAS_DAESolver

cp ../BONMIN_MINLPSolver/nlpsolver.h    ${BUILD_DIR}/usr/include/daetools/BONMIN_MINLPSolver

# Static libraries
cp ../release/libcdaeCore.a                ${BUILD_DIR}/usr/${LIB}/daetools
cp ../release/libcdaeActivity.a            ${BUILD_DIR}/usr/${LIB}/daetools
cp ../release/libcdaeDataReporting.a       ${BUILD_DIR}/usr/${LIB}/daetools
cp ../release/libcdaeIDAS_DAESolver.a      ${BUILD_DIR}/usr/${LIB}/daetools
cp ../release/libcdaeBONMIN_MINLPSolver.a  ${BUILD_DIR}/usr/${LIB}/daetools
cp ${IDAS}/lib/libsundials_idas.a          ${BUILD_DIR}/usr/${LIB}/daetools
cp ${IDAS}/lib/libsundials_nvecserial.a    ${BUILD_DIR}/usr/${LIB}/daetools

# Trilinos Amesos libraries
if [ -e ${TRILINOS}/libamesos.so ]; then
  cp ${TRILINOS}/libamesos.so         ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libepetra.so         ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libtriutils.so       ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libgaleri.so         ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libepetraext.so      ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libzoltan.so         ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libsimpi.so          ${BUILD_DIR}/usr/${LIB}
  cp ${TRILINOS}/libteuchos.so        ${BUILD_DIR}/usr/${LIB}
fi

# Bonmin libraries
if [ -d ${BONMIN}/lib ]; then
  cp -d ${BONMIN}/lib/*.so*            ${BUILD_DIR}/usr/${LIB}
fi

# Change permissions and strip .so libraries (Trilinos Amesos and Bonmin)
chmod -x ${BUILD_DIR}/usr/${LIB}/*.so* 
chmod -x ${BUILD_DIR}/usr/${LIB}/daetools/*.a* 
find ${BUILD_DIR}/usr/${LIB} -name \*.so* | xargs strip 

# Shortcuts
mkdir ${BUILD_DIR}/usr/share/applications

daePlotter_DESKTOP=${BUILD_DIR}/usr/share/applications/daetools-daePlotter.desktop
echo "[Desktop Entry]"                                 > ${daePlotter_DESKTOP}
#echo "Encoding=UTF-8"                                >> ${daePlotter_DESKTOP}
#echo "Version 1.0"                                   >> ${daePlotter_DESKTOP}
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
echo "[Desktop Entry]"                             > ${daeExamples_DESKTOP}
#echo "Encoding=UTF-8"                            >> ${daeExamples_DESKTOP}
#echo "Version 1.0"                               >> ${daeExamples_DESKTOP}
echo "Name=DAE Tools Examples"                    >> ${daeExamples_DESKTOP}
echo "GenericName=DAE Tools Examples"             >> ${daeExamples_DESKTOP}
echo "Comment=DAE Tools Examples"                 >> ${daeExamples_DESKTOP}
echo "Categories=GNOME;Development;"              >> ${daeExamples_DESKTOP}
echo "Exec=/usr/bin/daeexamples"                  >> ${daeExamples_DESKTOP}
echo "Icon=${ICON}"                               >> ${daeExamples_DESKTOP}
echo "Terminal=false"                             >> ${daeExamples_DESKTOP}
echo "Type=Application"                           >> ${daeExamples_DESKTOP}
echo "StartupNotify=true"                         >> ${daeExamples_DESKTOP}


if [ ${PCKG_TYPE} = "tgz" ]; then
  cd ${BUILD_DIR}
  tar -czvf ${TGZ} *
  cd ..

  cp ${BUILD_DIR}/${TGZ} ${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}_${ARCH}_${BOOST}.tar.gz 

elif [ ${PCKG_TYPE} = "deb" ]; then
  mkdir ${BUILD_DIR}/DEBIAN

  CONTROL=${BUILD_DIR}/DEBIAN/control
  echo "Package: ${PACKAGE_NAME} "                                                                   > ${CONTROL}
  echo "Version: ${VER_MAJOR}.${VER_MINOR}-${VER_BUILD} "                                           >> ${CONTROL}
  echo "Architecture: ${ARCH} "                                                                     >> ${CONTROL}
  echo "Section: math "                                                                             >> ${CONTROL}
  echo "Priority: optional "                                                                        >> ${CONTROL}
  echo "Installed-Size: 11,700 "                                                                    >> ${CONTROL}
  echo "Maintainer: Dragan Nikolic <dnikolic@daetools.com> "                                        >> ${CONTROL}
  if [ ${DISTRO} = "Debian-5" ]; then
    echo "Depends: python2.5, libboost1.35-dev, python-qt4, python-numpy, python-matplotlib, libc6" >> ${CONTROL}
  else
    echo "Depends: python2.6, libboost-all-dev, python-qt4, python-numpy, python-matplotlib, libc6" >> ${CONTROL}
  fi
  echo "Description: A cross-platform equation-oriented process modelling software. "               >> ${CONTROL}
  echo " DAE Tool is a cross-platform equation-oriented process modelling software. "               >> ${CONTROL}
  echo " This package includes pyDAE and cDAE modules. "                                            >> ${CONTROL}
  echo "Suggests: mayavi2, libsuitesparse-dev, libsuperlu3, liblapack3gf "                          >> ${CONTROL}
  echo "Replaces: libtrilinos"                                                                      >> ${CONTROL}
  echo "Conflicts: libtrilinos"                                                                     >> ${CONTROL}
  echo "Homepage: http://www.daetools.com "                                                         >> ${CONTROL}
 
  CONFFILES=${BUILD_DIR}/DEBIAN/conffiles
  echo "/etc/daetools/daetools.cfg"   > ${CONFFILES}
  echo "/etc/daetools/bonmin.cfg"    >> ${CONFFILES}

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

  SHLIBS=${BUILD_DIR}/DEBIAN/shlibs
  echo "libbonmin 0 libonmin.so.0 (>= 0:0.0.0)"                                           > ${SHLIBS}
  echo "libCbc 0 libCbc.so.0 (>= 0:0.0.0)"                                               >> ${SHLIBS}
  echo "libCbcSolver 0 libCbcSolver.so.0 (>= 0:0.0.0)"                                   >> ${SHLIBS}
  echo "libCgl 0 libCgl.so.0 (>= 0:0.0.0)"                                               >> ${SHLIBS}
  echo "libClp 0 libClp.so.0 (>= 0:0.0.0)"                                               >> ${SHLIBS}
  echo "libCoinUtils 0 libCoinUtils.so.0 (>= 0:0.0.0)"                                   >> ${SHLIBS}
  echo "libipopt 0 libipopt.so.0 (>= 0:0.0.0)"                                           >> ${SHLIBS}
  echo "libOsiCbc 0 libOsiCbc.so.0 (>= 0:0.0.0)"                                         >> ${SHLIBS}
  echo "libOsiClp 0 libOsiClp.so.0 (>= 0:0.0.0)"                                         >> ${SHLIBS}
  echo "libOsi 0 libOsi.so.0 (>= 0:0.0.0)"                                               >> ${SHLIBS}

  fakeroot dpkg -b ${BUILD_DIR}

elif [ ${PCKG_TYPE} = "rpm" ]; then
  cd ${BUILD_DIR}
  tar -czvf ${TGZ} *
  cd ..

  SPEC=dae-tools.spec
  echo "%define is_mandrake %(test -e /etc/mandrake-release && echo 1 || echo 0)"   > ${SPEC}
  echo "%define is_suse %(test -e /etc/SuSE-release && echo 1 || echo 0) "         >> ${SPEC}
  echo "%define is_fedora %(test -e /etc/fedora-release && echo 1 || echo 0) "     >> ${SPEC}

  echo "Summary: DAE Tools: A cross-platform equation-oriented process modelling software (pyDAE and cDAE modules). " >> ${SPEC}
  echo "Name: ${PACKAGE_NAME}"                                                      >> ${SPEC}
  echo "Version: ${VER_MAJOR}.${VER_MINOR}"                                         >> ${SPEC}
  echo "Release: ${VER_BUILD}"                                                      >> ${SPEC}
  echo "Packager:  Dragan Nikolic dnikolic@daetools.com"                            >> ${SPEC}
  echo "License: GNU GPL v3"                                                        >> ${SPEC}
  echo "URL: www.daetools.com"                                                      >> ${SPEC}
  echo "Provides: ${PACKAGE_NAME}"                                                  >> ${SPEC}
  echo "Requires: boost-devel, PyQt4, numpy, python-matplotlib "                    >> ${SPEC}
  echo "ExclusiveArch: ${ARCH_RPM}"                                                 >> ${SPEC}
  echo "Group: Development/Tools"                                                   >> ${SPEC}

  echo "%description"                                                               >> ${SPEC}
  echo "DAE Tools: A cross-platform equation-oriented process modelling software (pyDAE and cDAE modules). " >> ${SPEC}

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
  cp ~/rpmbuild/RPMS/${ARCH_RPM}/${RPM} ${INSTALLATIONS_DIR}/${PACKAGE_NAME}_${VER_MAJOR}.${VER_MINOR}-${VER_BUILD}_${ARCH}_${DISTRO}.rpm

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
