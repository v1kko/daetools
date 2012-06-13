#!/usr/bin/env python
import os, sys, platform, shutil

if platform.system() == 'Windows':
    sys_drive = os.environ['SYSTEMDRIVE']
    config_dir = os.path.realpath( os.path.join(sys_drive, '\\daetools') )
    #print 'config_dir = ', config_dir

    sm_dir = get_special_folder_path('CSIDL_COMMON_STARTMENU')
    shortcuts_dir = os.path.join(sm_dir, 'DAE Tools')
    
    if 'install' in sys.argv[1]:
        try:
            os.mkdir(config_dir)
        except Exception as e:
            print(e)

        try:
            shutil.move(os.path.join(sys.prefix, 'daetools.cfg'), config_dir)
        except Exception as e:
            print(e)
        try:
            shutil.move(os.path.join(sys.prefix, 'bonmin.cfg'), config_dir)
        except Exception as e:
            print(e)

        try:
            os.mkdir(shortcuts_dir)
        except Exception as e:
            print(e)
        
        try:
            dae_plotter = os.path.join(shortcuts_dir, 'daePlotter.lnk')
            args = '-c \"from daetools.daePlotter import daeStartPlotter; daeStartPlotter()\"'
            create_shortcut('pythonw', 'daePlotter', dae_plotter, args)
        except Exception as e:
            print(e)
        
        try:
            dae_examples = os.path.join(shortcuts_dir, 'daeExamples.lnk')
            args = '-c \"from daetools.examples.daeRunExamples import daeRunExamples; daeRunExamples()\"'
            create_shortcut('pythonw', 'daeExamples', dae_examples, args)
        except Exception as e:
            print(e)
        
    elif 'remove' in sys.argv[1]:
        shutil.rmtree(shortcuts_dir)
        
