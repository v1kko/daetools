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
import tutorial_che_1, tutorial_che_2, tutorial_che_3, tutorial_che_7
import tutorial_dealii_1

# Global variables (arguments for the generateSimulation function)
directory                   = None
py_simulation_file          = None
callable_object_name        = 'create_simulation_for_cosimulation'
additional_files            = []
localsAsOutputs             = True

# Script files to run all FMUs with the ComplianceChecker
run_fmuCheck_all_fmus_bat    = None
run_fmuCheck_all_fmus_sh     = None
compare_ref_and_cc_solutions = None

def formatDescription(simulation):
    description = simulation.m.Description.split('\n')
    description = '\n    '.join(description)
    return description

def exportFMU(simulation, log):
    global directory
    global py_simulation_file
    global callable_object_name
    global additional_files
    global localsAsOutputs
    global run_fmuCheck_all_fmus_bat
    global run_fmuCheck_all_fmus_sh
    
    # Create options for the FMI code generator and output files
    simulationClassName    = simulation.__class__.__name__
    relativeTolerance      = simulation.DAESolver.RelativeTolerance
    calculateSensitivities = simulation.CalculateSensitivities
    t_stop                 = simulation.TimeHorizon
    fmu_name               = simulation.m.GetStrippedName()
    numSteps               = int(simulation.TimeHorizon / simulation.ReportingInterval)
    if calculateSensitivities:
        arguments = '%s, relativeTolerance=%e, calculateSensitivities=%s' % (simulationClassName, relativeTolerance, calculateSensitivities)
    else:
        arguments = '%s, relativeTolerance=%e' % (simulationClassName, relativeTolerance)
    
    # Write shell scripts for executing the generated fmu with the ComplianceChecker
    #  a) GNU/Linux shell script
    #  b) Windows batch file
    FMUName_cc_sh  = './fmuCheck.linux64 -e %s_cc.log -o %s_cc.csv -n %d -s %.3f -l 5 %s.fmu' % (fmu_name, fmu_name, numSteps, t_stop, fmu_name) 
    FMUName_cc_bat = 'fmuCheck.win32.exe -e %s_cc.log -o %s_cc.csv -n %d -s %.3f -l 5 %s.fmu' % (fmu_name, fmu_name, numSteps, t_stop, fmu_name)

    f = open(os.path.join(directory, '%s_cc.sh' % fmu_name), "w")
    f.write('#!/bin/bash\n')
    f.write('# -*- coding: utf-8 -*-\n\n')
    f.write(FMUName_cc_sh)
    f.close()

    f = open(os.path.join(directory, '%s_cc.bat' % fmu_name), "w")
    f.write(FMUName_cc_bat)
    f.close()
    
    # Add the generated commands to the files for running all FMUs with the ComplianceChecker
    run_fmuCheck_all_fmus_bat.write('echo \"Executing %s...\"\n' % FMUName_cc_bat)
    run_fmuCheck_all_fmus_bat.write(FMUName_cc_bat + '\n\n')
    run_fmuCheck_all_fmus_sh.write('echo \"Executing %s...\"\n' % FMUName_cc_sh)
    run_fmuCheck_all_fmus_sh.write (FMUName_cc_sh  + '\n\n')
    
    # Write the options to create the reference output
    f = open(os.path.join(directory, '%s_ref.opt' % fmu_name), "wb")
    csv_f = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
    csv_f.writerow(['StartTime', 0.0])
    csv_f.writerow(['StopTime',  simulation.TimeHorizon])
    csv_f.writerow(['StepSize',  0.0])
    csv_f.writerow(['RelTol',    relativeTolerance])
    f.close()

    # Write the ReadMe.txt file
    f = open(os.path.join(directory, '%s-ReadMe.txt' % fmu_name), "w")
    f.write('Model Description:\n')
    f.write('    %s\n\n' % formatDescription(simulation))
    f.write('Compiler:\n')
    f.write('    Microsoft Visual C++ 2015, gcc 6.3\n\n')
    f.write('Available platforms:\n')
    f.write('    win32, linux64\n\n')
    f.write('Notes:\n')
    f.write('    \n')
    f.write('Contact:\n')
    f.write('    contact@daetools.com\n')
    f.close()
    
    # Generate the FMU file
    cg = daeCodeGenerator_FMI()
    cg.generateSimulation(simulation, 
                          directory            = directory, 
                          py_simulation_file   = py_simulation_file, 
                          callable_object_name = callable_object_name,
                          arguments            = arguments, 
                          additional_files     = additional_files,
                          localsAsOutputs      = localsAsOutputs)

def open_csv(csv_filename):
    # Open solution files and return variable names and their values.

    # Get the variable names from the.csv file using Python csv module.
    f = open(csv_filename, 'r')
    csv_f = csv.reader(f, delimiter = ',', quotechar = '"')
    names = {}
    header = csv_f.next()
    for index, name in enumerate(header):
        names[name] = index
        
    # Return to the beginning of the file and get the variable values. 
    f.seek(0,0)
    values = numpy.loadtxt(f, delimiter=',', skiprows=1, unpack=True)
    
    return names, values
    
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
    
    f = open(out_file, 'wb')
    csv_f = csv.writer(f, delimiter = ',', quotechar = '"')
    for varName, error in sorted(E.items()):
        csv_f.writerow(['\"%s\"' % varName, '%.5e' % error])
    f.close()
    
def exportSimulationFromTutorial(tutorial_module):
    # In Python 2.7 __file__ points to 'file.pyc' - correct it to 'file.py'
    global py_simulation_file
    py_simulation_file = tutorial_module.__file__
    if py_simulation_file.endswith('.pyc'):
        py_simulation_file = py_simulation_file[:-1]
        
    # To compare with the cc solution we need to remove duplicate times
    csv_filename = os.path.basename(py_simulation_file)
    csv_filename = os.path.splitext(csv_filename)[0]
    csv_cc_path  = os.path.join(directory, '%s_cc.csv'  % csv_filename)
    csv_ref_path = os.path.join(directory, '%s_ref.csv' % csv_filename)
    datareporter = daeCSVFileDataReporter()
    datareporter.Connect(csv_ref_path, csv_filename)
    
    # Run the simulation and export the .fmu file
    tutorial_module.run(generate_code_fn = exportFMU, 
                        datareporter     = datareporter,
                        stopAtModelDiscontinuity        = eDoNotStopAtDiscontinuity,
                        reportDataAroundDiscontinuities = False)
    
    compare_ref_and_cc_solutions.write('    compare_solutions(\'%s\', \'%s\', \'%s\')\n' % (os.path.basename(csv_ref_path), 
                                                                                            os.path.basename(csv_cc_path),
                                                                                            '%s-norm-error.csv' % csv_filename))
    
def generateFMUs():
    global directory
    global py_simulation_file
    global callable_object_name
    global additional_files
    global localsAsOutputs
    
    if not os.path.isdir(directory):
        os.makedirs(directory)

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
    directory = os.path.abspath(sys.argv[1])
    
    compare_ref_and_cc_solutions = open(os.path.join(directory, 'compare_ref_and_cc_solutions.py'), "wb")
    compare_ref_and_cc_solutions.write('#!/bin/bash\n')
    compare_ref_and_cc_solutions.write('# -*- coding: utf-8 -*-\n')
    compare_ref_and_cc_solutions.write('import os, sys, numpy, csv\n\n')
    compare_ref_and_cc_solutions.write(inspect.getsource(open_csv) + '\n')
    compare_ref_and_cc_solutions.write(inspect.getsource(compare_solutions) + '\n')
    compare_ref_and_cc_solutions.write('if __name__ == "__main__":\n')
   
    run_fmuCheck_all_fmus_bat = open(os.path.join(directory, 'run_fmuCheck_all_fmus-win32.bat'),  "wb")
    run_fmuCheck_all_fmus_sh  = open(os.path.join(directory, 'run_fmuCheck_all_fmus-linux64.sh'), "wb")
    run_fmuCheck_all_fmus_sh.write('#!/bin/bash\n')
    run_fmuCheck_all_fmus_sh.write('# -*- coding: utf-8 -*-\n\n')
    run_fmuCheck_all_fmus_sh.write('set -e\n\n')
        
    # Generate .fmu files
    generateFMUs()
