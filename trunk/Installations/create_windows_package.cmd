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
SET IDAS=%TRUNK%\idas-1.0.0\build
SET BONMIN=%TRUNK%\bonmin\build

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


REM Python modules
mkdir pyAmdACML
mkdir pyIntelMKL
mkdir pyIntelPardiso
mkdir pyTrilinosAmesos

echo on

cd %TRUNK%\release
copy pyCore.pyd                     %ROOT%\pyDAE
copy pyActivity.pyd                 %ROOT%\pyDAE
copy pyDataReporting.pyd            %ROOT%\pyDAE
copy pyIDAS.pyd                     %ROOT%\pyDAE
copy pyBONMIN.pyd                   %ROOT%\pyDAE
REM copy pyIPOPT.pyd                %ROOT%\pyDAE
REM copy IPOPT39.dll                %ROOT%\pyDAE

copy pyAmdACML.pyd                  %ROOT%\pyAmdACML
copy pyIntelMKL.pyd                 %ROOT%\pyIntelMKL
copy pyIntelPardiso.pyd             %ROOT%\pyIntelPardiso
copy pyTrilinosAmesos.pyd           %ROOT%\pyTrilinosAmesos

copy boost_python-vc90-mt-1_43.dll  %ROOT%\pyDAE

cd ..\python-files

REM Python files
copy daeLogs.py                  %ROOT%\pyDAE
copy WebView_ui.py               %ROOT%\pyDAE
copy WebViewDialog.py            %ROOT%\pyDAE

copy daetools__init__.py         %ROOT%\__init__.py
copy pyDAE__init__.py            %ROOT%\pyDAE\__init__.py
copy pyAmdACML__init__.py        %ROOT%\pyAmdACML\__init__.py
copy pyIntelMKL__init__.py       %ROOT%\pyIntelMKL\__init__.py
copy pyIntelPardiso__init__.py   %ROOT%\pyIntelPardiso\__init__.py
copy pyTrilinosAmesos__init__.py %ROOT%\pyTrilinosAmesos\__init__.py

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
copy *.png              %ROOT%\examples
copy *.init             %ROOT%\examples
copy daeRunExamples.py  %ROOT%\examples\daeRunExamples.pyw
cd images
copy *.*                %ROOT%\examples\images
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

REM Config
cd %TRUNK%
copy daetools.cfg  %ROOT%
copy bonmin.cfg    %ROOT% 

cd %INSTALL%
rem ECHO cd C:\Python%PYTHON_VERSION%\Lib\site-packages\%PACKAGE_NAME%\daePlotter > %ROOT%\daeplotter.cmd
rem ECHO daePlotter.pyw >> %ROOT%\daeplotter.cmd

cd %INSTALL%
ECHO import sys > setup.py
ECHO from distutils.core import setup >> setup.py
ECHO setup(name='%PACKAGE_NAME%',  >> setup.py
ECHO       version='%VERSION%',  >> setup.py
ECHO       description='DAE Tools',  >> setup.py
ECHO       long_description='DAE Tools: A cross-platform equation-oriented process modelling software (pyDAE modules).',  >> setup.py
ECHO       author='Dragan Nikolic',  >> setup.py
ECHO       author_email='dnikolic@daetools.com',  >> setup.py
ECHO       url='http:\\www.daetools.com',  >> setup.py
ECHO       license='GNU GPL v3',  >> setup.py
ECHO       platforms='%PLATFORM%',  >> setup.py
ECHO       packages=['%PACKAGE_NAME%'],  >> setup.py
ECHO       package_dir={'%PACKAGE_NAME%': '%DEST%'},  >> setup.py
ECHO       package_data={'%DEST%': ['*.*', 'pyDAE/*.*', 'examples/*.*', 'docs/*.*', 'docs/images/*.*', 'docs/api_ref/*.*', 'daeSimulator/*.*', 'daeSimulator/images/*.*', 'daePlotter/*.*', 'daePlotter/images/*.*', 'pyAmdACML/*.*', 'pyIntelMKL/*.*', 'pyLapack/*.*', 'pyIntelPardiso/*.*', 'pyAtlas/*.*', 'pyTrilinosAmesos/*.*']} >> setup.py
ECHO       )  >> setup.py

SET EXE=%PACKAGE_NAME%_%VER_MAJOR%.%VER_MINOR%.%VER_BUILD%_%PLATFORM%_%OS%_python%PYTHON_VERSION%.exe

"c:\Program Files\NSIS\makensis.exe" daetools.nsi
copy daetools.exe %EXE%
del daetools.exe

rem python setup.py bdist --format=%PCKG_TYPE%
rem copy %INSTALL%\dist\%EXEs% %EXEd%

rem SET EXEs=%PACKAGE_NAME%-%VER_MAJOR%.%VER_MINOR%.%VER_BUILD%.%PLATFORM%_python%PYTHON_VERSION%.exe
rem SET EXEd=%PACKAGE_NAME%_%VER_MAJOR%.%VER_MINOR%.%VER_BUILD%_%PLATFORM%_python%PYTHON_VERSION%.exe

rem rd /q /s build
rem rd /q /s dist

del setup.py
rd /q /s %ROOT%

cd %INSTALL%
