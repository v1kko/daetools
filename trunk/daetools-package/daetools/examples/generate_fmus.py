#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           generate_fmus.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
import os, sys, csv, inspect
from daetools.pyDAE import *
from daetools.code_generators.fmi import daeCodeGenerator_FMI
from daetools.pyDAE.data_reporters import daeCSVFileDataReporter

import whats_the_time
import tutorial1, tutorial4, tutorial5, tutorial14, tutorial15
import tutorial_che_1, tutorial_che_2, tutorial_che_3, tutorial_che_7, tutorial_che_9
import tutorial_dealii_1

# Global variables (arguments for the generateSimulation function)
base_directory        = None
fmu_name              = None
fmu_directory         = None
py_simulation_file    = None
callable_object_name  = 'run'
arguments             = 'initializeAndReturn=True'
additional_files      = []
localsAsOutputs       = True
add_xml_stylesheet    = True
useWebService         = True

# Script files to run all FMUs with the ComplianceChecker
run_fmuCheck_all_fmus_win32_bat  = None
run_fmuCheck_all_fmus_win64_bat  = None
run_fmuCheck_all_fmus_linux64_sh = None
compare_ref_and_cc_solutions     = None

def formatDescription(simulation):
    description = simulation.m.Description.split('\n')
    description = '\n    '.join(description)
    return description

def exportFMU(simulation, log):
    global fmu_name
    global fmu_directory
    global py_simulation_file
    global callable_object_name
    global arguments
    global additional_files
    global localsAsOutputs
    global add_xml_stylesheet
    global useWebService
    global run_fmuCheck_all_fmus_win32_bat
    global run_fmuCheck_all_fmus_win64_bat
    global run_fmuCheck_all_fmus_linux64_sh
       
    # Create options for the FMI code generator and output files
    simulationClassName    = simulation.__class__.__name__
    relativeTolerance      = simulation.DAESolver.RelativeTolerance
    calculateSensitivities = simulation.CalculateSensitivities
    t_stop                 = simulation.TimeHorizon
    numSteps               = int(simulation.TimeHorizon / simulation.ReportingInterval)
    fmu_directory_basename = os.path.basename(fmu_directory)
    fmu_relative_location  = os.path.join(fmu_directory_basename, fmu_name)
    
    # Write shell scripts for executing the generated fmu with the ComplianceChecker
    #  a) GNU/Linux shell script
    #  b) Windows batch file
    FMUName_cc_sh  = '$FMU_CHECK_BIN -e %s_cc.log -o %s_cc.csv -n %d -s %.3f -l 5 %s.fmu' % (fmu_name, fmu_name, numSteps, t_stop, fmu_name) 
    FMUName_cc_bat = 'start /wait %%FMU_CHECK_BIN%% -e %s_cc.log -o %s_cc.csv -n %d -s %.3f -l 5 %s.fmu' % (fmu_name, fmu_name, numSteps, t_stop, fmu_name)

    fmu_cc_sh = '%s_cc.sh' % fmu_name
    f = open(os.path.join(fmu_directory, fmu_cc_sh), "w")
    f.write('#!/bin/bash\n')
    f.write('# -*- coding: utf-8 -*-\n')
    #f.write('set -e\n\n')
    f.write(FMUName_cc_sh)
    f.close()

    fmu_cc_bat = '%s_cc.bat' % fmu_name
    f = open(os.path.join(fmu_directory, fmu_cc_bat), "w")
    f.write(FMUName_cc_bat)
    f.close()
    
    # Add the generated commands to the files for running all FMUs with the ComplianceChecker
    fmu_cc_sh_rel = os.path.join(fmu_directory_basename, fmu_cc_sh)
    run_fmuCheck_all_fmus_linux64_sh.write('echo \"Executing: sh %s...\"\n' % fmu_cc_sh_rel)
    run_fmuCheck_all_fmus_linux64_sh.write ('cd %s\n' % fmu_name)
    run_fmuCheck_all_fmus_linux64_sh.write ('sh ' + fmu_cc_sh  + '\n')
    run_fmuCheck_all_fmus_linux64_sh.write ('cd ..\n\n')
    
    fmu_cc_bat_rel = os.path.join(fmu_directory_basename, fmu_cc_bat)
    run_fmuCheck_all_fmus_win32_bat.write('echo \"Executing: %s...\"\n' % fmu_cc_bat_rel)
    run_fmuCheck_all_fmus_win32_bat.write ('cd %s\n' % fmu_name)
    run_fmuCheck_all_fmus_win32_bat.write('call ' + fmu_cc_bat + '\n')
    run_fmuCheck_all_fmus_win32_bat.write ('cd ..\n\n')

    fmu_cc_bat_rel = os.path.join(fmu_directory_basename, fmu_cc_bat)
    run_fmuCheck_all_fmus_win64_bat.write('echo \"Executing: %s...\"\n' % fmu_cc_bat_rel)
    run_fmuCheck_all_fmus_win64_bat.write ('cd %s\n' % fmu_name)
    run_fmuCheck_all_fmus_win64_bat.write('call ' + fmu_cc_bat + '\n')
    run_fmuCheck_all_fmus_win64_bat.write ('cd ..\n\n')
    
    # Write the options to create the reference output
    f = open(os.path.join(fmu_directory, '%s_ref.opt' % fmu_name), "w")
    csv_f = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
    csv_f.writerow(['StartTime', 0.0])
    csv_f.writerow(['StopTime',  simulation.TimeHorizon])
    csv_f.writerow(['StepSize',  0.0])
    csv_f.writerow(['RelTol',    relativeTolerance])
    f.close()

    # Write the ReadMe.txt file
    f = open(os.path.join(fmu_directory, 'ReadMe.txt'), "w")
    f.write('Model Description:\n')
    f.write('    %s\n\n' % formatDescription(simulation))
    f.write('Compiler:\n')
    f.write('    Microsoft Visual C++ 2015, gcc 6.3\n\n')
    f.write('Available platforms:\n')
    f.write('    win32, win64, linux64\n\n')
    f.write('Notes:\n')
    f.write('    \n')
    f.write('Contact:\n')
    f.write('    contact@daetools.com\n')
    f.close()
    
    # Generate the FMU file
    cg = daeCodeGenerator_FMI()
    cg.generateSimulation(simulation, 
                          directory            = fmu_directory, 
                          py_simulation_file   = py_simulation_file, 
                          callable_object_name = callable_object_name,
                          arguments            = arguments, 
                          additional_files     = additional_files,
                          localsAsOutputs      = localsAsOutputs,
                          add_xml_stylesheet   = add_xml_stylesheet,
                          useWebService        = useWebService)

def open_csv(csv_filename):
    # Open solution files and return variable names and their values.

    # Get the variable names from the.csv file using Python csv module.
    f = open(csv_filename, 'r')
    header = f.readline()
    names = header.split('","')
    dictNames = {}
    for index, name in enumerate(names):
        name = name.strip('\"\n\r')
        dictNames[name] = index

    # Return to the beginning of the file and get the variable values. 
    values = numpy.loadtxt(csv_filename, delimiter=',', skiprows=1, unpack=True)
    
    return dictNames, values
    
def compare_solutions(csv_ref_path, csv_cc_path, out_file):
    # Generates normalised global errors from the reference and the ComplianceChecker values.
    
    variables_ref, solution_ref = open_csv(csv_ref_path)
    variables_cc, solution_cc   = open_csv(csv_cc_path)
    
    E = {}
    for varName, index_cc in variables_cc.items():
        index_ref = variables_ref[varName]
        ref_solution = solution_ref[index_ref]
        cc_solution  = solution_cc[index_cc]
        n = len(cc_solution)
        E[varName] = numpy.sqrt((1.0/n) * numpy.sum((ref_solution-cc_solution)**2))
    
    maxE = numpy.max( list(E.values()) )
    print('%s: max normalised error = %.3e' % (out_file, maxE))
    
    f = open(out_file, 'w')
    for varName, error in sorted(E.items()):
        f.write('\"%s\",%.5e\n' % (varName, error))
    f.close()
    
def exportSimulationFromTutorial(tutorial_module):
    # In Python 2.7 __file__ points to 'file.pyc' - correct it to 'file.py'
    global py_simulation_file
    global base_directory
    global fmu_name
    global fmu_directory
    py_simulation_file = tutorial_module.__file__
    if py_simulation_file.endswith('.pyc'):
        py_simulation_file = py_simulation_file[:-1]
        
    fmu_name      = os.path.basename(py_simulation_file)
    fmu_name      = os.path.splitext(fmu_name)[0]
    fmu_directory = os.path.join(base_directory, fmu_name)
    if not os.path.isdir(fmu_directory):
        os.makedirs(fmu_directory)

    # To compare with the cc solution we need to remove duplicate times
    csv_filename = fmu_name
    csv_cc_path  = os.path.join(fmu_directory, '%s_cc.csv'  % csv_filename)
    csv_ref_path = os.path.join(fmu_directory, '%s_ref.csv' % csv_filename)
    datareporter = daeCSVFileDataReporter()
    datareporter.Connect(csv_ref_path, csv_filename)
    
    # Run the simulation and export the .fmu file
    tutorial_module.run(generate_code_fn = exportFMU, 
                        datareporter     = datareporter,
                        stopAtModelDiscontinuity        = eDoNotStopAtDiscontinuity,
                        reportDataAroundDiscontinuities = False)
    
    fmu_directory_basename = os.path.basename(fmu_directory)
    csv_ref_rel_path    = 'os.path.join(\"%s\",\"%s\")' % (fmu_directory_basename, os.path.basename(csv_ref_path))
    csv_cc_rel_path     = 'os.path.join(\"%s\",\"%s\")' % (fmu_directory_basename, os.path.basename(csv_cc_path))
    norm_error_rel_path = 'os.path.join(\"%s\",\"%s\")' % (fmu_directory_basename, '%s-norm-error.csv' % csv_filename)
    compare_ref_and_cc_solutions.write('    compare_solutions(%s, %s, %s)\n' % (csv_ref_rel_path, 
                                                                                csv_cc_rel_path,
                                                                                norm_error_rel_path))
    
def generateFMUs():
    global base_directory
    global fmu_directory
    global py_simulation_file
    global callable_object_name
    global additional_files
    global localsAsOutputs

    # Basic tutorials
    exportSimulationFromTutorial(whats_the_time)
    exportSimulationFromTutorial(tutorial1)
    exportSimulationFromTutorial(tutorial4)
    exportSimulationFromTutorial(tutorial5)
    exportSimulationFromTutorial(tutorial14)
    exportSimulationFromTutorial(tutorial15)

    # Chemical Engineering examples
    exportSimulationFromTutorial(tutorial_che_1)
    exportSimulationFromTutorial(tutorial_che_2)
    exportSimulationFromTutorial(tutorial_che_3)
    exportSimulationFromTutorial(tutorial_che_7)
    exportSimulationFromTutorial(tutorial_che_9)
    
    # Finite Element tutorials (deal.II)
    mesh_filename    = 'step-49.msh'
    meshes_dir       = os.path.join(os.path.dirname(os.path.abspath(tutorial_dealii_1.__file__)), 'meshes')
    additional_files = [ (os.path.join(meshes_dir, mesh_filename), 'meshes/%s' % mesh_filename) ]
    exportSimulationFromTutorial(tutorial_dealii_1)
    additional_files = [] # reset the list for the use with the other simulations

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python generate_fmus.py directory')
        sys.exit()
    
    # Set the output directory
    base_directory = os.path.abspath(sys.argv[1])
    if not os.path.isdir(base_directory):
        os.makedirs(base_directory)
    
    compare_ref_and_cc_solutions = open(os.path.join(base_directory, 'compare_ref_and_cc_solutions.py'), "w")
    compare_ref_and_cc_solutions.write('#!/bin/bash\n')
    compare_ref_and_cc_solutions.write('# -*- coding: utf-8 -*-\n')
    compare_ref_and_cc_solutions.write('import os, sys, numpy, csv\n\n')
    compare_ref_and_cc_solutions.write(inspect.getsource(open_csv) + '\n')
    compare_ref_and_cc_solutions.write(inspect.getsource(compare_solutions) + '\n')
    compare_ref_and_cc_solutions.write('if __name__ == "__main__":\n')
   
    run_fmuCheck_all_fmus_win32_bat = open(os.path.join(base_directory, 'run_fmuCheck_all_fmus-win32.bat'),  "w")
    run_fmuCheck_all_fmus_win32_bat.write('set FMU_CHECK_BIN=../fmuCheck.win32.exe\n\n')

    run_fmuCheck_all_fmus_win64_bat = open(os.path.join(base_directory, 'run_fmuCheck_all_fmus-win64.bat'),  "w")
    run_fmuCheck_all_fmus_win64_bat.write('set FMU_CHECK_BIN=../fmuCheck.win64.exe\n\n')
    
    run_fmuCheck_all_fmus_linux64_sh  = open(os.path.join(base_directory, 'run_fmuCheck_all_fmus-linux64.sh'), "w")
    run_fmuCheck_all_fmus_linux64_sh.write('#!/bin/bash\n')
    run_fmuCheck_all_fmus_linux64_sh.write('# -*- coding: utf-8 -*-\n\n')
    run_fmuCheck_all_fmus_linux64_sh.write('set -e\n')
    #run_fmuCheck_all_fmus_linux64_sh.write('set -x\n\n')
    run_fmuCheck_all_fmus_linux64_sh.write('export FMU_CHECK_BIN=../fmuCheck.linux64\n\n')
        
    # Generate .fmu files
    generateFMUs()
