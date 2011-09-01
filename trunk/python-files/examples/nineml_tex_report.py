#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys, subprocess
from time import localtime, strftime, time
from daetools.pyDAE.parser import ExpressionParser
from daetools.pyDAE import *

class texCommand:
    def __init__(self, startCommand, endCommand):
        self.startCommand = startCommand
        self.endCommand   = endCommand

    def __call__(self, content):
        return self.startCommand + content + self.endCommand

tex_itemize  = texCommand('\\begin{itemize}\n', '\\end{itemize}\n')
tex_verbatim = texCommand(' \\begin{verbatim}', '\\end{verbatim} ')

class nineml_tex_report:
    def __init__(self, ninemlComponent, texTemplate, texOutputFile):
        self.ninemlComponent = ninemlComponent
        self.texOutputFile   = texOutputFile
        self.texTemplate     = texTemplate
        self.parser = ExpressionParser()

        self.tf = open(texTemplate, 'r')
        self.template = ''.join(self.tf.readlines())
        self.tf.close()
        self.of = open(texOutputFile, 'w')

        self.begin_itemize = '\\begin{itemize}\n'
        self.item          = '\\item '
        self.end_itemize   = '\\end{itemize}\n\n'

        self.content = []
        # 1) Create parameters
        if self.ninemlComponent.parameters:
            self.content.append('\\subsection*{Parameters}\n\n')
            self.content.append(self.begin_itemize)
            for param in self.ninemlComponent.parameters:
                tex = self.item + tex_verbatim(param.name) + '\n'
                self.content.append(tex)
            self.content.append(self.end_itemize)
            self.content.append('\n')

        # 2) Create state-variables (diff. variables)
        if self.ninemlComponent.state_variables:
            self.content.append('\\subsection*{State-Variables}\n\n')
            self.content.append(self.begin_itemize)
            for var in self.ninemlComponent.state_variables:
                tex = self.item + tex_verbatim(var.name) + '\n'
                self.content.append(tex)
            self.content.append(self.end_itemize)
            self.content.append('\n')

        # 3) Create alias variables (algebraic)
        if self.ninemlComponent.aliases_map:
            self.content.append('\\subsection*{Aliases}\n\n')
            for alias in self.ninemlComponent.aliases:
                tex = '${0} = {1}$\n\n'.format(alias.lhs, self.parser.parse_to_latex(alias.rhs))
                self.content.append(tex)
            self.content.append('\n')

        # 4) Create analog-ports and reduce-ports
        if self.ninemlComponent.analog_ports:
            self.content.append('\\subsection*{Analog Ports}\n\n')
            self.content.append(self.begin_itemize)
            for analog_port in self.ninemlComponent.analog_ports:
                tex = self.item + tex_verbatim(analog_port.name + ' (' + analog_port.mode + ')') + '\n'
                self.content.append(tex)
            self.content.append(self.end_itemize)
            self.content.append('\n')

        # 5) Create event-ports
        if self.ninemlComponent.event_ports:
            self.content.append('\\subsection*{Event ports}\n\n')
            self.content.append(self.begin_itemize)
            for event_port in self.ninemlComponent.event_ports:
                tex = self.item + tex_verbatim(event_port.name + ' (' + event_port.mode + ')') + '\n'
                self.content.append(tex)
            self.content.append(self.end_itemize)
            self.content.append('\n')

        # 6) Create sub-nodes
        #self.ninemlSubComponents = []
        #for name, subcomponent in self.ninemlComponent.subnodes.items():
        #    self.ninemlSubComponents.append( nineml_daetools_bridge(name, subcomponent, self) )

        # 7) Create port connections
        if self.ninemlComponent.portconnections:
            self.content.append('\\subsection*{Port Connections}\n\n')
            self.content.append(self.begin_itemize)
            for port_connection in self.ninemlComponent.portconnections:
                portFrom = port_connection[0]
                portTo   = port_connection[1]
                tex = self.item + ''
                #self.content.append(tex)
            self.content.append(self.end_itemize)
            self.content.append('\n')

        # 8) Create regimes
        if self.ninemlComponent.regimes:
            self.content.append('\\subsection*{Regimes}\n\n')
            for regime in self.ninemlComponent.regimes:
                counter = 0
                tex = ''
                # 8a) Create time derivatives
                for time_deriv in regime.time_derivatives:
                    if counter != 0:
                        tex += ' \\\\ '
                    tex += '\\frac{{d{0}}}{{dt}} = {1}'.format(time_deriv.dependent_variable, self.parser.parse_to_latex(time_deriv.rhs))
                    counter += 1
                
                # 8b) Create on_condition actions
                for on_condition in regime.on_conditions:
                    tex += ' \\\\ \\mbox{If } ' + self.parser.parse_to_latex(on_condition.trigger.rhs) + '\mbox{:}'

                    if on_condition.target_regime.name != '':
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(on_condition.target_regime.name)

                    for state_assignment in on_condition.state_assignments:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, self.parser.parse_to_latex(state_assignment.rhs))

                    for event_output in on_condition.event_outputs:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name)
                        
                # 8c) Create on_event actions
                for on_event in regime.on_events:
                    tex += ' \\\\ \\mbox{On } ' + on_event.src_port_name + '\mbox{:}'

                    if on_event.target_regime.name != '':
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(on_event.target_regime.name)

                    for state_assignment in on_event.state_assignments:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, self.parser.parse_to_latex(state_assignment.rhs))

                    for event_output in on_event.event_outputs:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name)

                tex = '${0} = \\begin{{cases}} {1} \\end{{cases}}$\n'.format(regime.name, tex)
                tex += '\\newline \n'
                self.content.append(tex)
                
            self.content.append('\n')

        self.template = self.template.replace('###MODEL_NAME###', self.ninemlComponent.name)
        self.template = self.template.replace('###MODEL_SPECIFICATION###', ''.join(self.content))
        self.template = self.template.replace('###TESTS###', '')
        self.template = self.template.replace('###APPENDIXES###', '')
        self.of.write(self.template)
        self.of.close()

        if subprocess.call(['pdflatex', self.texOutputFile], shell=False) != 0:
            raise RuntimeError('Call to pdflatex failed!')
        subprocess.call(['evince', self.texOutputFile.replace('.tex', '.pdf')], shell=False)

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
    coba_iaf_base = TestableComponent('hierachical_iaf_1coba')
    coba_iaf = coba_iaf_base()

    destexhe_ampa_base = TestableComponent('destexhe_ampa')
    destexhe_ampa = destexhe_ampa_base()

    coba = nineml_tex_report(coba_iaf.subnodes['cobaExcit'], 'nineml-tex-template.tex', 'cobaExcit.tex')
    iaf  = nineml_tex_report(coba_iaf.subnodes['iaf'],       'nineml-tex-template.tex', 'iaf.tex')
    ampa  = nineml_tex_report(destexhe_ampa, 'nineml-tex-template.tex', 'destexhe_ampa.tex')


