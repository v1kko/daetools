import os, sys, tempfile, inspect
from daetools.pyDAE import *
from daetools.code_generators.fmi import daeCodeGenerator_FMI

tmp_folder = tempfile.mkdtemp(prefix = 'daetools-code_generator-fmi-')
base_dir = os.path.dirname(__file__)

def generateFMU(className, timeHorizon, reportingInterval):
    cg = daeCodeGenerator_FMI()
    module_path = os.path.join(base_dir, inspect.getmodule(className).__name__ + '.py')
    args = 'simTutorial, %f, %f' % (timeHorizon, reportingInterval)
    simulation = instantiate_simulation_by_name(className, timeHorizon, reportingInterval)
    cg.generateSimulation(simulation, tmp_folder, module_path, 'instantiate_simulation_by_name', args, [])

#import whats_the_time
#generateFMU(whats_the_time.simTutorial, 1000, 10)

import tutorial1
generateFMU(tutorial1.simTutorial, 1000, 10)

import tutorial2
generateFMU(tutorial2.simTutorial, 1000, 10)

import tutorial3
#generateFMU(tutorial3.simTutorial, 200, 5)

import tutorial4
generateFMU(tutorial4.simTutorial, 500, 10)

import tutorial5
generateFMU(tutorial5.simTutorial, 500, 2)

import tutorial6
#generateFMU(tutorial6.simTutorial, 100, 10)

#import tutorial7 (DO NOT RUN)
#generateFMU(tutorial7.simTutorial, 1000, 10)

import tutorial8
generateFMU(tutorial8.simTutorial, 100, 10)

import tutorial9
generateFMU(tutorial9.simTutorial, 1000, 10)

import tutorial10
generateFMU(tutorial10.simTutorial, 500, 10)

import tutorial11
generateFMU(tutorial11.simTutorial, 1000, 10)

import tutorial12
generateFMU(tutorial12.simTutorial, 1000, 10)

import tutorial13
generateFMU(tutorial13.simTutorial, 500, 2)

import tutorial14
generateFMU(tutorial14.simTutorial, 500, 0.5)

import tutorial15
generateFMU(tutorial15.simTutorial, 500, 10)

#import tutorial16 (DO NOT RUN)
#generateFMU(tutorial16.simTutorial, 1000, 10)

#import tutorial17 (DO NOT RUN)
#generateFMU(tutorial17.simTutorial, 1000, 10)

import tutorial18
generateFMU(tutorial18.simTutorial, 60, 5)

#import tutorial19 (DO NOT RUN)
#generateFMU(tutorial19.simTutorial, 1000, 10)

import tutorial20
generateFMU(tutorial20.simTutorial, 100, 10)

import tutorial_dealii_1
generateFMU(tutorial_dealii_1.simTutorial, 500, 10)

import tutorial_dealii_2
generateFMU(tutorial_dealii_2.simTutorial, 60*60, 60)
