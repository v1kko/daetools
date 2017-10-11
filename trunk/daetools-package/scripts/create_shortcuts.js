var oWS = WScript.CreateObject("WScript.Shell");
var fso = new ActiveXObject("Scripting.FileSystemObject");

var pythonw          = WScript.Arguments.Item(0)
var python_version   = WScript.Arguments.Item(1)
var daetools_version = WScript.Arguments.Item(2)

var daetools_suffix =  "_"+ daetools_version + "_py" + python_version;

strStartmenu = oWS.SpecialFolders("StartMenu") + "\\DAE Tools" + daetools_suffix ;
strDesktop = oWS.SpecialFolders("Desktop");
try
{
  fso.CreateFolder(strStartmenu);
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strStartmenu + "\\DAE Tools Examples" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.examples.run_examples';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strStartmenu + "\\DAE Tools Plotter" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.dae_plotter.plotter';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strStartmenu + "\\DAE Tools Web Service" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.dae_simulator.daetools_ws';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strStartmenu + "\\DAE Tools FMI Web Service" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.dae_simulator.daetools_fmi_ws';
  oLink.Save();
}
catch(e)
{
}

/*
try
{
  var oLink = oWS.CreateShortcut(strDesktop + "\\daeExamples" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.examples.run_examples';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strDesktop + "\\daePlotter" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -m daetools.dae_plotter.plotter';
  oLink.Save();
}
catch(e)
{
}
*/
