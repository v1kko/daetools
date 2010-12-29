; The name of the installer
Name "DAE Tools"

; The file to write
OutFile "daetools.exe"

; The default installation directory
InstallDir C:\Python26\Lib\site-packages\daetools

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
  File    "daetools\*.*"
  File /r "daetools\pyDAE\*.*"
  File /r "daetools\pyAmdACML\*.*"
  File /r "daetools\pyIntelMKL\*.*"
  File /r "daetools\pyLapack\*.*"
  File /r "daetools\pyTrilinosAmesos\*.*"
  File /r "daetools\pyIntelPardiso\*.*"
  File /r "daetools\daePlotter\*.*"
  File /r "daetools\daeSimulator\*.*"
  File /r "daetools\docs\*.*"
  File /r "daetools\examples\*.*"

 ; Headers and libs
  CreateDirectory c:\daetools
  CreateDirectory c:\daetools\include
  CreateDirectory c:\daetools\lib

  SetOutPath c:\daetools\include
  File /r "daetools\cDAE\*.*"

  ; Config file
  CopyFiles $INSTDIR\daetools_cfg  c:\daetools\daetools.cfg
  CopyFiles $INSTDIR\bonmin_cfg    c:\daetools\bonmin.cfg
  Delete $INSTDIR\daetools_cfg
  Delete $INSTDIR\bonmin_cfg

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

  SetOutPath $INSTDIR\docs
  CreateShortCut "$SMPROGRAMS\DAE Tools\Documentation.lnk" "$INSTDIR\docs\documentation.html" "" "$INSTDIR\docs\documentation.html" 0

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
