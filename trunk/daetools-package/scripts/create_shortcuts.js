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
  var oLink = oWS.CreateShortcut(strStartmenu + "\\daeExamples" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -c "from daetools.examples.run_examples import daeRunExamples; daeRunExamples()"';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strStartmenu + "\\daePlotter" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -c "from daetools.dae_plotter.plotter import daeStartPlotter; daeStartPlotter()"';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strDesktop + "\\daeExamples" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -c "from daetools.examples.run_examples import daeRunExamples; daeRunExamples()"';
  oLink.Save();
}
catch(e)
{
}

try
{
  var oLink = oWS.CreateShortcut(strDesktop + "\\daePlotter" + daetools_suffix +".lnk");
  oLink.TargetPath = pythonw;
  oLink.Arguments = ' -c "from daetools.dae_plotter.plotter import daeStartPlotter; daeStartPlotter()"';
  oLink.Save();
}
catch(e)
{
}

