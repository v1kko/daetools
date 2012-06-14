#!python
import os, sys, platform, shutil

if platform.system() == 'Windows':
    messages = []
    sys_drive = os.environ['SYSTEMDRIVE']
    config_dir = os.path.realpath( os.path.join(sys_drive, '\\daetools') )

    sm_dir = get_special_folder_path('CSIDL_COMMON_STARTMENU')
    shortcuts_dir = os.path.join(sm_dir, 'DAE Tools')

    if 'install' in sys.argv[1]:
        try:
            os.mkdir(config_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            shutil.move(os.path.join(sys.prefix, 'daetools.cfg'), config_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            shutil.move(os.path.join(sys.prefix, 'bonmin.cfg'), config_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            os.mkdir(shortcuts_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            #dae_plotter = os.path.join(shortcuts_dir, 'daePlotter.lnk')
            #args = '-c \"from daetools.daePlotter import daeStartPlotter; daeStartPlotter()\"'
            #create_shortcut('pythonw.exe', 'daePlotter', dae_plotter, args)
            shutil.move(os.path.join(sys.prefix, 'daePlotter.lnk'), shortcuts_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            #dae_examples = os.path.join(shortcuts_dir, 'daeExamples.lnk')
            #args = '-c \"from daetools.examples.daeRunExamples import daeRunExamples; daeRunExamples()\"'
            #create_shortcut('pythonw.exe', 'daeExamples', dae_examples, args)
            shutil.move(os.path.join(sys.prefix, 'daeExamples.lnk'), shortcuts_dir)
        except Exception as e:
            messages.append(str(e))

        try:
            print '\n'.join(messages)
            sys.stdout.flush()
        except Exception as e:
            pass

    elif 'remove' in sys.argv[1]:
        shutil.rmtree(shortcuts_dir)

