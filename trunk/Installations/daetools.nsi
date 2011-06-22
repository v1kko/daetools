; The name of the installer
Name "DAE Tools"

; The file to write
OutFile "daetools.exe"

; The default installation directory
; Python version: PYTHON_VERSION=`$PYTHON -c "import sys; print (\"%d.%d\" % (sys.version_info[0], sys.version_info[1]))"`
InstallDir C:\Python27\Lib\site-packages\daetools

; Registry key to check for directory (so if you install again, it will
; overwrite the old one automatically)
InstallDirRegKey HKLM "Software\NSIS_daetools" "Install_Dir"

; Request application privileges for Windows Vista
RequestExecutionLevel admin

;--------------------------------

; Pages

Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------

; The stuff to install
Section "daetools (required)"

  SectionIn RO

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR
  File /r   "daetools\*.*"

  CopyFiles $INSTDIR\pyDAE\boost_python-vc90-mt-1_46_1.dll  $INSTDIR\solvers

  ; Config file
  CreateDirectory c:\daetools
  CopyFiles $INSTDIR\daetools.cfg  c:\daetools
  CopyFiles $INSTDIR\bonmin.cfg    c:\daetools
  Delete $INSTDIR\daetools.cfg
  Delete $INSTDIR\bonmin.cfg

  ; Write the installation path into the registry
  WriteRegStr HKLM SOFTWARE\NSIS_daetools "Install_Dir" "$INSTDIR"

  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\daetools" "DisplayName" "DAE Tools"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\daetools" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\daetools" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\daetools" "NoRepair" 1
  WriteUninstaller "uninstall.exe"

SectionEnd

; Optional section (can be disabled by the user)
Section "Start Menu Shortcuts"

  CreateDirectory "$SMPROGRAMS\DAE Tools"

  SetOutPath $INSTDIR\daePlotter
  CreateShortCut "$SMPROGRAMS\DAE Tools\daePlotter.lnk" "$INSTDIR\daePlotter\daePlotter.pyw" "50000" "$INSTDIR\daePlotter\daePlotter.pyw" 0

  SetOutPath $INSTDIR\examples
  CreateShortCut "$SMPROGRAMS\DAE Tools\DAE Tools Examples.lnk" "$INSTDIR\examples\daeRunExamples.pyw" "" "$INSTDIR\examples\daeRunExamples.pyw" 0

  CreateShortCut "$SMPROGRAMS\DAE Tools\Uninstall.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\uninstall.exe" 0

SectionEnd

;--------------------------------

; Uninstaller

Section "Uninstall"

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\daetools"
  DeleteRegKey HKLM SOFTWARE\NSIS_daetools

  ; Remove directories used
  RMDir /r "$SMPROGRAMS\DAE Tools"
  RMDir /r "$INSTDIR"
  RMDir /r c:\daetools

SectionEnd
