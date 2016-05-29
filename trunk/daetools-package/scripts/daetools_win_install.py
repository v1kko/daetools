#!/usr/bin/env python
"""********************************************************************************
                            daetools_win_install.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, sys, platform, shutil

daetools_version = '1.4.0'
python_major = str(sys.version_info[0])
python_minor = str(sys.version_info[1])

messages = []

def install():
    try:
        if platform.system() == 'Windows':
            sys_drive = os.environ['SYSTEMDRIVE']
            config_dir = os.path.join(sys_drive, os.path.sep, 'daetools')
            messages.append('Config files directory: %s' % config_dir)
            
            pyw_executable = os.path.join(sys.prefix, "pythonw.exe")
            
            try:
                if not os.path.exists(config_dir):
                    os.mkdir(config_dir)
                directory_created(config_dir)
            except Exception as e:
                messages.append(str(e))

            try:
                f = os.path.join(config_dir, 'daetools.cfg')
                if os.path.exists(f):
                    os.remove(f)
                shutil.copy(os.path.join(sys.prefix, 'daetools.cfg'), config_dir)
                file_created(f)
            except Exception as e:
                messages.append(str(e))

            try:
                f = os.path.join(config_dir, 'bonmin.cfg')
                if os.path.exists(f):
                    os.remove(f)
                shutil.copy(os.path.join(sys.prefix, 'bonmin.cfg'), config_dir)
                file_created(f)
            except Exception as e:
                messages.append(str(e))

            try:
                script_folder = os.path.dirname(os.path.abspath(__file__))
            except:
                script_folder = os.path.abspath(os.path.dirname(sys.argv[0]))
            pythonw_path     = os.path.join(sys.prefix, 'pythonw.exe')
            python_version   = '%s.%s' % (python_major, python_minor)
            cmd = 'cscript %s\create_shortcuts.js %s %s %s' % (script_folder, pythonw_path, python_version, daetools_version)
            messages.append(cmd)
            os.system(cmd)
    finally:
        try:
            print('DAE Tools install messages:')
            print('\n'.join(messages))
            #sys.stdout.flush()
        except Exception as e:
           print(str(e))            
        
        #try:
        #    f = open(os.path.join(os.path.expanduser("~"), 'daetools-install.log'), 'w')
        #    for line in messages:
        #        f.write(line + '\n')
        #    f.close()
        #except Exception as e:
        #    print(str(e))
            
if len(sys.argv) > 1:
    if 'install' in sys.argv[1]:
        install()
    elif 'remove' in sys.argv[1]:
        pass
else:            
    install()                    
