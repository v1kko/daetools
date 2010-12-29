REM                              %1        %2        %3        %4          %5      %6
REM   create_windows_package.cmd ver_major ver_minor ver_build platform    os      python version
REM [ create_windows_package.cmd 1         0         0         win32       winxp   26

@ECHO off
SET PACKAGE_NAME=daetools
SET VER_MAJOR=%1
SET VER_MINOR=%2
SET VER_BUILD=%3
SET OS=%5
SET PYTHON_VERSION=%6
SET DEST=daetools
SET VERSION=%VER_MAJOR%.%VER_MINOR%.%VER_BUILD%
SET PCKG_TYPE=wininst
SET PLATFORM=%4
SET TRUNK=c:\Data\daetools\trunk
SET INSTALL=%TRUNK%\Installations
SET ROOT=%INSTALL%\%DEST%
IDAS=..\idas-1.0.0\build

ECHO " "
ECHO "***************************"
ECHO "*      SETTINGS:          *"
ECHO "***************************"
ECHO "  Version: " %VERSION%
ECHO "  Platform: " %PLATFORM%
ECHO "  Package type: " %PCKG_TYPE%
ECHO "  Trunk: " %TRUNK%
ECHO "  Install: " %INSTALL%
ECHO "  Root: " %ROOT%
ECHO "***************************"
ECHO " "

cd %INSTALL%
mkdir %DEST%
cd %DEST%

mkdir examples
mkdir docs
mkdir daePlotter
mkdir daeSimulator
mkdir cDAE
mkdir pyDAE

cd examples
mkdir images
cd ..

cd docs
mkdir images
mkdir api_ref
cd ..

cd daePlotter
mkdir images
cd..

cd daeSimulator
mkdir images
cd..


cd cDAE
mkdir lib
mkdir include
cd include
mkdir Core
mkdir Activity
mkdir DataReporting
mkdir IDAS_DAESolver
cd ..
cd ..

REM Python modules
mkdir pyAmdACML
mkdir pyIntelMKL
mkdir pyIntelPardiso

echo on

cd %TRUNK%\release
copy pyCore.pyd                     %ROOT%\pyDAE\pyCore.pyd
copy pyActivity.pyd                 %ROOT%\pyDAE\pyActivity.pyd
copy pyDataReporting.pyd            %ROOT%\pyDAE\pyDataReporting.pyd
copy pyIDAS.pyd                     %ROOT%\pyDAE\pyIDAS.pyd
copy pyBONMIN.pyd                   %ROOT%\pyDAE\pyBONMIN.pyd
copy IPOPT39.dll                    %ROOT%\pyDAE\IPOPT39.dll

copy pyAmdACML.pyd                  %ROOT%\pyAmdACML\pyAmdACML.pyd
copy pyIntelMKL.pyd                 %ROOT%\pyIntelMKL\pyIntelMKL.pyd
copy pyIntelPardiso.pyd             %ROOT%\pyIntelPardiso\pyIntelPardiso.pyd

copy boost_python-vc90-mt-1_43.dll  %ROOT%\pyDAE\boost_python-vc90-mt-1_43.dll
copy boost_python-vc90-mt-1_43.dll  %ROOT%\pyAmdACML\boost_python-vc90-mt-1_43.dll
copy boost_python-vc90-mt-1_43.dll  %ROOT%\pyIntelMKL\boost_python-vc90-mt-1_43.dll
copy boost_python-vc90-mt-1_43.dll  %ROOT%\pyIntelPardiso\boost_python-vc90-mt-1_43.dll

cd ..\python-files

REM Python files
copy daetools__init__.py         %ROOT%\__init__.py
copy pyDAE__init__.py            %ROOT%\pyDAE\__init__.py
copy daeLogs.py                  %ROOT%\pyDAE\daeLogs.py
copy pyAmdACML__init__.py        %ROOT%\pyAmdACML\__init__.py
copy pyIntelMKL__init__.py       %ROOT%\pyIntelMKL\__init__.py
copy pyIntelPardiso__init__.py   %ROOT%\pyIntelPardiso\__init__.py
copy WebView_ui.py               %ROOT%\pyDAE\WebView_ui.py
copy WebViewDialog.py            %ROOT%\pyDAE\WebViewDialog.py

REM daePlotter
cd daePlotter
copy *.py           %ROOT%\daePlotter
copy daePlotter.py  %ROOT%\daePlotter\daePlotter.pyw
cd images
copy *.*            %ROOT%\daePlotter\images
cd ..
cd ..

REM daeSimulator
cd daeSimulator
copy *.py           %ROOT%\daeSimulator
cd images
copy *.*            %ROOT%\daeSimulator\images
cd ..
cd ..

REM Examples and Tutorials
cd examples
copy *.*                %ROOT%\examples
copy *.css              %ROOT%\examples
copy *.html             %ROOT%\examples
copy *.xsl              %ROOT%\examples
copy *.xml              %ROOT%\examples
copy *.py               %ROOT%\examples
copy *.out              %ROOT%\examples
copy *.png              %ROOT%\examples
copy *.init             %ROOT%\examples
copy daeRunExamples.py  %ROOT%\examples\daeRunExamples.pyw
cd images
copy *.*               %ROOT%\examples\images
cd ..
cd ..

REM Website
cd ..\Website
copy *.html  %ROOT%\docs
del %ROOT%\docs\downloads.html

cd images
copy *.png    %ROOT%\docs\images
copy *.css    %ROOT%\docs\images
copy *.gif    %ROOT%\docs\images
copy *.jpg    %ROOT%\docs\images
cd ..\api_ref
copy *.html   %ROOT%\docs\api_ref
cd %ROOT%

REM Include
cd %TRUNK%
copy dae.h           %ROOT%\cDAE\include\dae.h
copy dae_develop.h   %ROOT%\cDAE\include\dae_develop.h

cd %TRUNK%\Core
copy definitions.h      %ROOT%\cDAE\include\Core\definitions.h
copy xmlfile.h          %ROOT%\cDAE\include\Core\xmlfile.h
copy helpers.h          %ROOT%\cDAE\include\Core\helpers.h
copy base_logging.h     %ROOT%\cDAE\include\Core\base_logging.h
copy macros.h           %ROOT%\cDAE\include\Core\macros.h
copy class_factory.h    %ROOT%\cDAE\include\Core\class_factory.h
copy coreimpl.h         %ROOT%\cDAE\include\Core\coreimpl.h

cd %TRUNK%\Activity
copy base_activities.h  %ROOT%\cDAE\include\Activity\base_activities.h
copy simulation.h       %ROOT%\cDAE\include\Activity\simulation.h

cd %TRUNK%\DataReporting
copy datareporters.h                    %ROOT%\cDAE\include\DataReporting\datareporters.h
copy base_data_reporters_receivers.h    %ROOT%\cDAE\include\DataReporting\base_data_reporters_receivers.h

cd %TRUNK%\IDAS_DAESolver
copy base_solvers.h     %ROOT%\cDAE\include\IDAS_DAESolver\base_solvers.h
copy ida_solver.h       %ROOT%\cDAE\include\IDAS_DAESolver\ida_solver.h

REM Lib
cd %TRUNK%\release
copy cdaeCore.lib                %ROOT%\cDAE\lib\cdaeCore.lib
copy cdaeActivity.lib            %ROOT%\cDAE\lib\cdaeActivity.lib
copy cdaeIDAS_DAESolver.lib      %ROOT%\cDAE\lib\cdaeIDAS_DAESolver.lib
copy cdaeDataReporting.lib       %ROOT%\cDAE\lib\cdaeDataReporting.lib
copy cdaeBONMIN_MINLPSolver.lib  %ROOT%\cDAE\lib\cdaeBONMIN_MINLPSolver.lib

copy %IDAS%\lib\sundials_idas.lib         %ROOT%\cDAE\lib\sundials_idas.lib
copy %IDAS%\lib\sundials_nvecserial.lib   %ROOT%\cDAE\lib\sundials_nvecserial.lib

REM Config
cd %TRUNK%
copy daetools.cfg  %ROOT%\daetools_cfg 
copy bonmin.cfg    %ROOT%\bonmin_cfg 

cd %INSTALL%
rem ECHO cd C:\Python%PYTHON_VERSION%\Lib\site-packages\%PACKAGE_NAME%\daePlotter > %ROOT%\daeplotter.cmd
rem ECHO daePlotter.pyw >> %ROOT%\daeplotter.cmd

cd %INSTALL%
ECHO import sys > setup.py
ECHO from distutils.core import setup >> setup.py
ECHO setup(name='%PACKAGE_NAME%',  >> setup.py
ECHO       version='%VERSION%',  >> setup.py
ECHO       description='DAE Tools',  >> setup.py
ECHO       long_description='DAE Tools: A cross-platform equation-oriented process modelling software (pyDAE and cDAE modules).',  >> setup.py
ECHO       author='Dragan Nikolic',  >> setup.py
ECHO       author_email='dnikolic@daetools.com',  >> setup.py
ECHO       url='http:\\www.daetools.com',  >> setup.py
ECHO       license='GNU GPL v3',  >> setup.py
ECHO       platforms='%PLATFORM%',  >> setup.py
ECHO       packages=['%PACKAGE_NAME%'],  >> setup.py
ECHO       package_dir={'%PACKAGE_NAME%': '%DEST%'},  >> setup.py
ECHO       package_data={'%DEST%': ['*.*', 'pyDAE/*.*', 'examples/*.*', 'docs/*.*', 'docs/images/*.*', 'docs/api_ref/*.*', 'daeSimulator/*.*', 'daeSimulator/images/*.*', 'daePlotter/*.*', 'daePlotter/images/*.*', 'cDAE/include/*.*', 'cDAE/include/Core/*.*', 'cDAE/include/DataReporters/*.*', 'cDAE/include/Simulation/*.*', 'cDAE/include/Solver/*.*', 'cDAE/lib/*.*', 'pyAmdACML/*.*', 'pyIntelMKL/*.*', 'pyLapack/*.*', 'pyIntelPardiso/*.*', 'pyAtlas/*.*', 'pyTrilinosAmesos/*.*']} >> setup.py
ECHO       )  >> setup.py

SET EXE=%PACKAGE_NAME%_%VER_MAJOR%.%VER_MINOR%-%VER_BUILD%_%PLATFORM%_%OS%_python%PYTHON_VERSION%.exe

"c:\Program Files\NSIS\makensis.exe" daetools.nsi
copy daetools.exe %EXE%
del daetools.exe

rem python setup.py bdist --format=%PCKG_TYPE%
rem copy %INSTALL%\dist\%EXEs% %EXEd%

rem SET EXEs=%PACKAGE_NAME%-%VER_MAJOR%.%VER_MINOR%.%VER_BUILD%.%PLATFORM%_python%PYTHON_VERSION%.exe
rem SET EXEd=%PACKAGE_NAME%_%VER_MAJOR%.%VER_MINOR%-%VER_BUILD%_%PLATFORM%_python%PYTHON_VERSION%.exe

rem rd /q /s build
rem rd /q /s dist

del setup.py
rd /q /s %ROOT%

cd %INSTALL%
