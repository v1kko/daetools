#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys, subprocess
from time import localtime, strftime, time
from daetools.pyDAE.expression_parser import ExpressionParser
from daetools.pyDAE.units_parser import UnitsParser
from daetools.pyDAE import *
from nineml_component_inspector import nineml_component_inspector

class texCommand:
    def __init__(self, startCommand, endCommand):
        self.startCommand = startCommand
        self.endCommand   = endCommand

    def __call__(self, content):
        return self.startCommand + content + self.endCommand

tex_itemize  = texCommand('\\begin{itemize}\n', '\\end{itemize}\n')
tex_verbatim = texCommand(' \\begin{verbatim}', '\\end{verbatim} ')

def createLatexReport(inspector, texTemplate, texOutputFile, find_files_dir = '.'):
    tf = open(texTemplate, 'r')
    template = ''.join(tf.readlines())
    tf.close()
    of = open(texOutputFile, 'w')
    content = inspector.generateLatexReport()

    comp_name = inspector.ninemlComponent.name.replace('_', '\\_')
    template  = template.replace('LOGO_PATH', find_files_dir)
    template  = template.replace('MODEL-NAME', comp_name)
    template  = template.replace('MODEL-SPECIFICATION', ''.join(content))
    template  = template.replace('TESTS', '')
    template  = template.replace('APPENDIXES', '')
    
    of.write(template)
    of.close()

def createPDF(texFile, outdir = None):
    if outdir:
        res = os.system('/usr/bin/pdflatex -output-directory {0} {1}'.format(outdir, texFile))
    else:
        res = os.system('/usr/bin/pdflatex {0}'.format(texFile))
        
    return res

"""
\begin{table}[placement=h]
    \caption{A normal caption}
    \begin{center}
      \begin{tabular}{ | l | l |}
    \hline
    Time, ms & Vm, mV \\ \hline
    0 & -50.0 \\
    1 & -51.0 \\
    2 & -52.0 \\
    3 & -53.0 \\
    4 & -54.0 \\
    5 & -55.0 \\
    \hline
      \end{tabular}
    \end{center}
\end{table}
"""

if __name__ == "__main__":
    nineml_component = TestableComponent('hierachical_iaf_1coba')()
    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component)
    createLatexReport(inspector, 'nineml-tex-template.tex', 'coba_iaf.tex')
    res = createPDF('coba_iaf.tex')
    subprocess.call(['evince', 'coba_iaf.pdf'], shell=False)

