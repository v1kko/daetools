"""
***********************************************************************************
                            cxx_mpi.py
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
import os, shutil, sys, numpy, math, traceback, pprint, struct, json
from daetools.pyDAE import *
from .formatter import daeExpressionFormatter
from .analyzer import daeCodeGeneratorAnalyzer
from .code_generator import daeCodeGenerator
try:
    import pygraphviz as pgv
except Exception as e:
    print(str(e))

class daeInterProcessCommGraph(object):
    def __init__(self, partitionData, statsData):
        # node is a Runtime Node object
        self.counter = 0
        self.graph   = pgv.AGraph(strict = False) # strict=False enables multi-graph
        
        self._processPEs(partitionData, statsData)
    
    def SaveGraph(self, filename, layout = 'circo'):
        # Layouts: dot|neato|circo|twopi|fdp|nop
        # The best layout is made by 'dot'
        self.graph.layout(prog = layout)
        self.graph.draw(filename)

    def _addNode(self, labelChild, shape='box', color='black', fontcolor='black'):
        child = str(self.counter)
        self.counter += 1
        
        self.graph.add_node(child, label = str(labelChild), shape = shape, color = color, fontcolor = fontcolor)
        
        return child

    def _addEdge(self, source, destination, label, color = 'gray50', fontcolor = 'black'):
        self.graph.add_edge(source, destination, label = label, color = color, fontcolor = fontcolor, dir = 'forward', arrowType = 'normal')

    def _processPEs(self, partitionData, statsData):
        #label = 'Neq = %d\lNeq/pe = %.1f\lNadj/pe = %.1f\l' % (totalEquations, averageEquations, averageAdjacent)
        #self._addNode(label, color='black', fontcolor='black')
        
        stats = statsData
        nodes = {}
        for pe, data in partitionData.items():
            #label = 'PE %d (%d : %d)\n%s %s %s %s %s' % (pe, data['Nequations'], data['Nadjacent'], equations[pe], ipcLoad[pe], ncsLoad[pe], flopsLoad[pe], nnzLoad[pe])
            labelFormat = 'PE %d (%d : %d)\ndev Neq = %+.2f%% \ldev Nad = %+.2f%% \ldev Ncs = %+.2f%% \ldev Nflr = %+.2f%% \ldev Nnz = %+.2f%%\ldev Nflj = %+.2f%%\l'
            label = labelFormat % (pe, 
                                   data['Nequations'], 
                                   data['Nadjacent'], 
                                   stats['Nequations'][pe], 
                                   stats['Nadjacent'][pe], 
                                   stats['Ncs'][pe], 
                                   stats['Nflops'][pe], 
                                   stats['Nnz'][pe], 
                                   stats['Nflops_jacobian'][pe])
            nodes[pe] = self._addNode(label, color='blue', fontcolor='blue')
            
        for pe, data in partitionData.items():
            node = nodes[pe]
            for pe_send_to, Nindexes in data['send_to'].items():
                node_send_to = nodes[pe_send_to]
                label        = '%d' % Nindexes
                self._addEdge(node, node_send_to, label)
    
class daeCodeGenerator_MPI(daeCodeGenerator):
    """
    Limitations:
     - DOFs in models are not supported (for the block indexes are not set uniformly)
     - STNs are not supported
    """
    def __init__(self):
        self.wrapperInstanceName     = ''
        self.topLevelModel           = None
        self.simulation              = None
        self.equationGenerationMode  = ''
        self.balancingConstraints    = []
        
        self.analyzer = daeCodeGeneratorAnalyzer()
        
        # MPI
        self.Npe = 0
        self.errors = []

    def generateSimulation(self, simulation, directory, Npe, balancingConstraints = [], unaryOperationsFlops = {}, binaryOperationsFlops = {}):
        # Returns partitioningData, statsData (deviations from average in %)
        if not simulation:
            raise RuntimeError('Invalid simulation object')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if Npe <= 0:
            raise RuntimeError('Invalid number of processing elements')
        
        self.simulation    = simulation
        self.topLevelModel = simulation.m
        
        for constraint in balancingConstraints:
            if constraint != 'Ncs' and constraint != 'Nflops' and constraint != 'Nnz' and constraint != 'Nflops_jacobian':
                raise RuntimeError('Invalid balancing constraint: %s' % constraint)
        self.balancingConstraints = balancingConstraints
        
        # wrapperInstanceName should not be stripped of illegal characters, since it is used to get relative names
        self.wrapperInstanceName = simulation.m.Name

        # Computational complexity of unary/binary mathematical operations (for METIS vweigths).
        # If operation does not exist in the dictionary 1 is assumed.
        """ 
        - Unary functions:
            eSign, eSqrt, eExp, eLog, eLn, eAbs, eCeil, eFloor, eErf
            eSin, eCos, eTan, eArcSin, eArcCos, eArcTan, eSinh, eCosh, eTanh, eArcSinh, eArcCosh, eArcTanh.
        - Binary functions:
            ePlus eMinus eMulti eDivide ePower eMin eMax eArcTan2
        """ 
        if not unaryOperationsFlops or len(unaryOperationsFlops) == 0:
            d_unaryFlops = {eSqrt : 6,
                            eExp  : 9}
        else:
            d_unaryFlops = unaryOperationsFlops
        if not binaryOperationsFlops or len(binaryOperationsFlops) == 0:
            d_binaryFlops = {eMulti  : 2,
                             eDivide : 4}
        else:
            d_binaryFlops = binaryOperationsFlops

        self.analyzer.analyzeSimulation(simulation, d_unaryFlops, d_binaryFlops)
        
        self.Npe = Npe

        log_filename = os.path.join(directory, 'code-generation.log')
        self.logf = open(log_filename, 'w')
        
        partitionData, statsData = self._generateRuntimeInformation(self.analyzer.runtimeInformation, directory)
        #print('partitionData')
        #print(json.dumps(partitionData, indent = 4))
        #print('statsData')
        #print(json.dumps(statsData, indent = 4))
        
        self.logf.close()
        
        if len(self.errors):
            raise RuntimeError('\n\n'.join(self.errors))
        
        stats          = statsData
        modelName      = self.wrapperInstanceName
        latexModelName = self.wrapperInstanceName.replace('_', '\\_')

        if len(self.balancingConstraints) == 0:
            filename_objectives = '%s-%dpe-edge_cut' % (modelName, Npe)
        else:
            objectives_s        = '_'.join(self.balancingConstraints)
            filename_objectives = '%s-%dpe-edge_cut-%s' % (modelName, Npe, objectives_s)
        
        try:
            ipcGraph = daeInterProcessCommGraph(partitionData, statsData)
            graph_filename = os.path.join(directory, 'partition_graph-%s.png' % filename_objectives)
            ipcGraph.SaveGraph(graph_filename)
        except Exception as e:
            print(str(e))            

        # Generate json file with DAE and LA solver/preconditioner options.
        solver_options = """{
    "OutputDirectory": "results",
    "DAESolver" : {
        "Library" : "Sundials",
        "Name": "IDAS",
        "Parameters": {
            "MaxOrd":             5,
            "MaxNumSteps":        500,
            "InitStep":           0.0,
            "MaxStep":            0.0,
            "MaxErrTestFails":    10,
            "MaxNonlinIters":     4,
            "MaxConvFails":       10,
            "NonlinConvCoef":     0.33,
            "SuppressAlg":        false,
            "NoInactiveRootWarn": false,
            "NonlinConvCoefIC":   0.0033,
            "MaxNumStepsIC":      5,
            "MaxNumJacsIC":       4,
            "MaxNumItersIC":      10,
            "LineSearchOffIC":    false
        }
    },
    "LinearSolver" : {
        "Library" : "Sundials",
        "Name": "gmres",
        "Parameters": {
            "kspace":            30,
            "EpsLin":            0.05,
            "JacTimesVecFn":     "DQ",
            "DQIncrementFactor": 1.0,
            "MaxRestarts":       5,
            "GSType":            "MODIFIED_GS"
        },
        "Preconditioner" : {
            "Library" : "Ifpack",
            "Name": "ILU",
            "Parameters": {
                "fact: level-of-fill":      3,
                "fact: relax value":        0.0,
                "fact: absolute threshold": 1e-5,
                "fact: relative threshold": 1.0
            }
        }
    }
}"""
        solver_options_filename = os.path.join(directory, 'solver_options.json')
        f = open(solver_options_filename, 'w')
        f.write(solver_options) #json.dumps(solver_options, indent = 4))
        f.close()

        # Generate Latex table
        caption_objs = 'none'
        if len(self.balancingConstraints) > 0:
            constraints = []
            for constraint in self.balancingConstraints:
                if constraint == 'Ncs':
                    constraints.append('$N_{cs}$')
                elif constraint == 'Nflops':
                    constraints.append('$N_{flops}$')
                elif constraint == 'Nnz':
                    constraints.append('$N_{nz}$')
                elif constraint == 'Nflops_jacobian':
                    constraints.append('$N_{flops\\_jacobian}$')
            caption_objs = ', '.join(constraints)
        
        label_objs = 'none'
        if len(self.balancingConstraints) > 0:
            label_objs = '-'.join(self.balancingConstraints)
        
        latexTable  = '\\begin{table}[h!] \n'
        latexTable += '\\centering \n'
        latexTable += '\\caption{Partitioning results ($N_{pe} = %d$, constraints: %s)} \n' % (Npe, caption_objs)
        latexTable += '\\label{table:PartitioningResults-%s-%dpe-%s} \n' % (modelName, Npe, label_objs)
        latexTable += '  \\begin{tabular}{lrrrrrrrr} \n'
        latexTable += '    \\hline \n'
        latexTable += '            PE &   $N_{eq}$ &  $N_{adj}$ &    $N_{eq}^{dev}$ &   $N_{adj}^{dev}$ &    $N_{cs}^{dev}$ & $N_{flops}^{dev}$ &    $N_{nz}^{dev}$ & $N_{flops\\_jacobian}^{dev}$ \\\\ \n'
        latexTable += '    \\hline \n'
        for pe, data in partitionData.items():
            latexTable += '    %10d & %10d & %10d & %17.2f & %17.2f & %17.2f & %17.2f & %17.2f & %27.2f \\\\ \n' % (pe, 
                                                                                                                    data['Nequations'], 
                                                                                                                    data['Nadjacent'], 
                                                                                                                    stats['Nequations'][pe], 
                                                                                                                    stats['Nadjacent'][pe], 
                                                                                                                    stats['Ncs'][pe], 
                                                                                                                    stats['Nflops'][pe], 
                                                                                                                    stats['Nnz'][pe], 
                                                                                                                    stats['Nflops_jacobian'][pe])
        latexTable += '    \\hline \n'
        latexTable += '    \\multicolumn{3}{r}{Max abs dev:    } & %17.2f & %17.2f & %17.2f & %17.2f & %17.2f & %27.2f \\\\ \n' % (numpy.max(numpy.abs(stats['Nequations'])), 
                                                                                                                                   numpy.max(numpy.abs(stats['Nadjacent'])), 
                                                                                                                                   numpy.max(numpy.abs(stats['Ncs'])), 
                                                                                                                                   numpy.max(numpy.abs(stats['Nflops'])), 
                                                                                                                                   numpy.max(numpy.abs(stats['Nnz'])), 
                                                                                                                                   numpy.max(numpy.abs(stats['Nflops_jacobian'])))
        latexTable += '    \\hline \n'
        latexTable += '  \\end{tabular} \n'
        latexTable += '\\end{table} \n'
        latexTable += '\n'
        latexTable += '\\begin{figure}[h!] \n'     
        latexTable += '    \\centering \n'
        latexTable += '    \\includegraphics[width=14cm]{%s} \n' % (os.path.basename(graph_filename))
        latexTable += '    \\caption{Partition graph ($N_{pe} = %d$, constraints: %s)} \n' % (Npe, caption_objs)
        latexTable += '    \\label{fig:PartitionGraph-%s-%dpe-%s} \n' % (modelName, Npe, label_objs)
        latexTable += '\\end{figure} \n'
        latexTable += '\n'
        """
        for pe in range(Npe):
            latexTable += '\\begin{figure}[h!] \n'     
            latexTable += '    \\centering \n'
            latexTable += '    \\includegraphics[width=3cm]{%s} \n' % ('incidence_matrix-%05d.png' % pe)
            latexTable += '\\end{figure} \n'
        latexTable += '\\caption{Incidence matrix ($N_{pe} = %d$, additional objectives: %s)} \n' % (Npe, caption_objs)
        latexTable += '\\label{fig:incidence_matrix-%s-%dpe-%s} \n' % (modelName, Npe, label_objs)
        """
        latexTable += '\\FloatBarrier \n'
        latexTable += '\\clearpage \n'
        #print(latexTable)
        
        latex_filename = os.path.join(directory, 'partition_stats-%s.tex' % filename_objectives)
        f = open(latex_filename, 'w')
        f.write(latexTable)
        f.close()

        # Generate CSV file
        csvTable = '   PE,         Neq,        Nadj,     Neq_dev,    Nadj_dev,     Ncs_dev,  Nflops_dev,     Nnz_dev, Nflops_jac_dev \n'
        for pe, data in partitionData.items():
            csvTable += '%5d, %11d, %11d, %11.2f, %11.2f, %11.2f, %11.2f, %11.2f, %14.2f\n' % (pe, 
                                                                                               data['Nequations'], 
                                                                                               data['Nadjacent'], 
                                                                                               stats['Nequations'][pe], 
                                                                                               stats['Nadjacent'][pe], 
                                                                                               stats['Ncs'][pe], 
                                                                                               stats['Nflops'][pe], 
                                                                                               stats['Nnz'][pe], 
                                                                                               stats['Nflops_jacobian'][pe])
        csvTable += '     ,            ,Max abs dev:, %11.2f, %11.2f, %11.2f, %11.2f, %11.2f, %14.2f\n' % (numpy.max(numpy.abs(stats['Nequations'])), 
                                                                                                           numpy.max(numpy.abs(stats['Nadjacent'])), 
                                                                                                           numpy.max(numpy.abs(stats['Ncs'])), 
                                                                                                           numpy.max(numpy.abs(stats['Nflops'])), 
                                                                                                           numpy.max(numpy.abs(stats['Nnz'])), 
                                                                                                           numpy.max(numpy.abs(stats['Nflops_jacobian'])))
        #print(csvTable)
        
        csv_filename = os.path.join(directory, 'partition_stats-%s.csv' % filename_objectives)
        f = open(csv_filename, 'w')
        f.write(csvTable)
        f.close()

        return partitionData, statsData
    
    def _generateRuntimeInformation(self, runtimeInformation, directory):
        # Returns part_results, stats_results
        Ntotal             = runtimeInformation['TotalNumberOfVariables']
        Neq                = runtimeInformation['NumberOfEquations']
        IDs                = runtimeInformation['IDs']
        dofs               = runtimeInformation['DOFs']
        initValues         = runtimeInformation['Values']
        initDerivatives    = runtimeInformation['TimeDerivatives']
        indexMappings      = runtimeInformation['IndexMappings']
        absoluteTolerances = runtimeInformation['AbsoluteTolerances']

        variableNames        = numpy.empty(Neq, dtype=object)
        blockIDs             = numpy.zeros(Neq, dtype=numpy.int32)
        blockInitValues      = numpy.zeros(Neq, dtype=numpy.float64)
        absTolerances        = numpy.zeros(Neq, dtype=numpy.float64)
        blockInitDerivatives = numpy.zeros(Neq, dtype=numpy.float64)
        variableNames[:]        = ['']  * Neq
        blockIDs[:]             = [-1]  * Neq
        blockInitValues[:]      = [-1.0]* Neq
        absTolerances[:]        = [1E-5]* Neq
        blockInitDerivatives[:] = [0.0] * Neq
        for oi, bi in list(indexMappings.items()):
            if IDs[oi] == cnAlgebraic:
               blockIDs[bi] = cnAlgebraic
            elif IDs[oi] == cnDifferential:
               blockIDs[bi] = cnDifferential

            if IDs[oi] != cnAssigned:
                blockInitValues[bi]      = initValues[oi]
                blockInitDerivatives[bi] = initDerivatives[oi]
                absTolerances[bi]        = absoluteTolerances[oi]
        
        for variable in runtimeInformation['Variables']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, variable['CanonicalName'])
            #formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            #name           = self.exprFormatter.flattenIdentifier(formattedName)
            numberOfPoints = variable['NumberOfPoints']
            units          = ('-' if (variable['Units'] == unit()) else str(variable['Units']))

            if numberOfPoints == 1:
                ID           = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value        = float(variable['Values'])   # numpy float
                overallIndex = variable['OverallIndex']
                fullName     = relativeName

                blockIndex = None
                if ID != cnAssigned:
                    blockIndex = indexMappings[overallIndex]
                    variableNames[blockIndex] = fullName

            else:
                domainsIndexesMap = variable['DomainsIndexesMap']
                for i in range(0, numberOfPoints):
                    domIndexes   = tuple(domainsIndexesMap[i])              # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    fullName     = relativeName + '(' + ','.join(str(di) for di in domIndexes) + ')'

                    blockIndex = None
                    if ID != cnAssigned:
                        blockIndex = indexMappings[overallIndex]
                        variableNames[blockIndex] = fullName
        
        interprocess_sync_map, equationBlockIndexes = self._partitionDAESystem(blockIDs, directory)
        
        # Generate partitioning results (to be returned by this function)
        part_results = {}
        for pe,data in interprocess_sync_map.items():
            pe_data = {}
            part_results[pe] = pe_data
            
            pe_data['Nequations'] = len(data['owned_indexes'])
            pe_data['Nadjacent']  = len(data['foreign_indexes'])
            
            send_to = {}
            for pe_send_to, indexes in data['send_to'].items():
                send_to[pe_send_to] = len(indexes)
            pe_data['send_to'] = send_to
            
            receive_from = {}
            for pe_receive_from, indexes in data['receive_from'].items():
                receive_from[pe_receive_from] = len(indexes)
            pe_data['receive_from'] = receive_from
                
            pe_data['Ncs']             = data['Ncs_load']
            pe_data['Nflops']          = data['FLOPS_load']
            pe_data['Nnz']             = data['NNZ_load']
            pe_data['Nflops_jacobian'] = data['FLOPS_Jacob_Load']
        
        # Statisticl data (deviations from average, in %)
        stats_results = self._generateStatistics(part_results)
                        
        #################################################
        # Generate:                                     #
        #  - runtime_data-%05d.bin                      #
        #  - incidence_matrix-%05d.png                  #
        #################################################
        # Requires interprocess_sync_map
        def packIntegerArray(arr):
            buff = struct.pack('i', len(arr))
            fmt = '%di' % len(arr)
            buff += struct.pack(fmt, *arr)
            return buff
        
        def packDoubleArray(arr):
            buff = struct.pack('i', len(arr))
            fmt = '%dd' % len(arr)
            buff += struct.pack(fmt, *arr)
            return buff
        
        def packStringArray(arr):
            buff = struct.pack('i', len(arr))
            for item in arr:
                bytes_item = str.encode(item, encoding = 'utf-8') # or ascii
                buff += struct.pack('i', len(bytes_item))
                fmt = '%ds' % len(bytes_item)
                buff += struct.pack(fmt, bytes_item)
            return buff
        
        for pe, pe_data in interprocess_sync_map.items():
            Neq_local       = pe_data['Nequations']
            owned_indexes   = pe_data['owned_indexes']
            
            init_values         = blockInitValues[owned_indexes]
            init_derivatives    = blockInitDerivatives[owned_indexes]
            absolute_tolerances = absTolerances[owned_indexes]
            ids                 = blockIDs[owned_indexes]
            variable_names      = variableNames[owned_indexes]
                        
            buff = bytes()
            buff += struct.pack('I', Ntotal)
            buff += struct.pack('I', Neq)
            buff += struct.pack('I', Neq_local)
            buff += struct.pack('I', len(dofs))
            buff += struct.pack('d', 0.0)
            buff += struct.pack('d', runtimeInformation['TimeHorizon'])
            buff += struct.pack('d', runtimeInformation['ReportingInterval'])
            buff += struct.pack('d', runtimeInformation['RelativeTolerance'])        
            buff += struct.pack('?', runtimeInformation['QuasiSteadyState'])
            
            buff += packDoubleArray(dofs)
            buff += packDoubleArray(init_values)
            buff += packDoubleArray(init_derivatives)
            buff += packDoubleArray(absolute_tolerances)
            buff += packIntegerArray(ids)
            buff += packStringArray(variable_names)
            
            filename = os.path.join(directory, 'runtime_data-%05d.bin' % pe)
            f = open(filename, 'wb')
            f.write(buff)
            f.close()
            
            # Incidence matrix images.
            try:
                import png
                Nequations_local = pe_data['Nequations']
                bi2bi_local      = pe_data['bi_to_bi_local']
                owned_indexes    = pe_data['owned_indexes']
                
                filename = os.path.join(directory, 'incidence_matrix-%05d.png' % pe)
                f = open(filename, 'wb')
                w = png.Writer(Nequations_local, Nequations_local, greyscale=True)
                pixels = numpy.zeros((Nequations_local, Nequations_local), dtype=numpy.int32)
                pixels += 255
                for row,ei in enumerate(owned_indexes):
                    eqn_bi_indexes = equationBlockIndexes[ei]
                    for bi in eqn_bi_indexes:
                        col = bi2bi_local[bi]
                        if col < Nequations_local: # skip foreign indexes
                            pixels[row,col] = 0
                w.write(f, pixels.tolist())
                f.close()
            except Exception as e:
                print(str(e))

        #################################################
        # Generate:                                     #
        #  - interprocess_comm_data-%05d.bin            #
        #  - model_equations-%05d.bin                   #
        #  - preconditioner_data-%05d.bin               #
        #################################################
        def packArray(arr):
            buff = struct.pack('i', len(arr))
            fmt = '%di' % len(arr)
            buff += struct.pack(fmt, *arr)
            return buff
        
        for pe, n_data in interprocess_sync_map.items():
            buff = bytes()

            buff += packArray(n_data['foreign_indexes'])
            
            dict_items = n_data['bi_to_bi_local'].items()
            buff += struct.pack('i', len(dict_items))
            for bi,bi_local in dict_items:
                buff += struct.pack('ii', bi, bi_local)

            dict_items = n_data['send_to'].items()
            buff += struct.pack('i', len(dict_items))
            for sti, st_data in dict_items:
                # Node index
                buff += struct.pack('i', sti)
                # Length and array of indexes
                buff += packArray(st_data)
            
            dict_items = n_data['receive_from'].items()
            buff += struct.pack('i', len(dict_items))
            for rfi, rf_data in dict_items:
                # Node index
                buff += struct.pack('i', rfi)
                # Length and array of indexes
                buff += packArray(rf_data)
            
            equation_indexes = n_data['owned_indexes']
            bi_to_bi_local   = n_data['bi_to_bi_local']
            filenameCS = os.path.join(directory, 'model_equations-%05d.bin' % pe)
            filenameJI = os.path.join(directory, 'jacobian_data-%05d.bin' % pe)
            self.simulation.ExportComputeStackStructs(filenameCS, filenameJI, equation_indexes, bi_to_bi_local)
            
            filename = os.path.join(directory, 'partition_data-%05d.bin' % pe)
            f = open(filename, 'wb')
            f.write(buff)
            f.close()
        
        return part_results, stats_results

    def _generateStatistics(self, partitionData):
        statistics = {}
        
        totalEquations      = 0
        totalAdjacent       = 0
        totalNcsLoad        = 0
        totalFLOPSLoad      = 0
        totalFLOPSJacobLoad = 0
        totalNNZLoad          = 0
        averageEquations      = 0
        averageAdjacent       = 0
        averageNcsLoad        = 0
        averageFLOPSLoad      = 0
        averageNNZLoad        = 0
        averageFLOPSJacobLoad = 0
        Npe             = len(partitionData)
        equations       = [0.0] * Npe
        ipcLoad         = [0.0] * Npe
        ncsLoad         = [0.0] * Npe
        flopsLoad       = [0.0] * Npe
        nnzLoad         = [0.0] * Npe
        flopsJacobLoad  = [0.0] * Npe
        for pe, data in partitionData.items():
            totalEquations      += data['Nequations']
            totalAdjacent       += data['Nadjacent']
            totalNcsLoad        += data['Ncs']
            totalFLOPSLoad      += data['Nflops']
            totalNNZLoad        += data['Nnz']
            totalFLOPSJacobLoad += data['Nflops_jacobian']

        averageEquations        = totalEquations        / len(partitionData)
        averageAdjacent         = totalAdjacent         / len(partitionData)
        averageNcsLoad          = totalNcsLoad          / len(partitionData)
        averageFLOPSLoad        = totalFLOPSLoad        / len(partitionData)
        averageNNZLoad          = totalNNZLoad          / len(partitionData)
        averageFLOPSJacobLoad   = totalFLOPSJacobLoad   / len(partitionData)

        statistics['Nequations']      = []
        statistics['Nadjacent']       = []
        statistics['Ncs']             = []
        statistics['Nflops']          = []
        statistics['Nnz']             = []
        statistics['Nflops_jacobian'] = []        
        for pe, data in partitionData.items():
            statistics['Nequations'].append(      (100.0 * (data['Nequations']      - averageEquations)      / averageEquations) )
            if averageAdjacent == 0:
                statistics['Nadjacent'].append(   0.0 )
            else:
                statistics['Nadjacent'].append(   (100.0 * (data['Nadjacent']       - averageAdjacent)       / averageAdjacent) )
            statistics['Ncs'].append(             (100.0 * (data['Ncs']             - averageNcsLoad)        / averageNcsLoad) )
            statistics['Nflops'].append(          (100.0 * (data['Nflops']          - averageFLOPSLoad)      / averageFLOPSLoad) )
            statistics['Nnz'].append(             (100.0 * (data['Nnz']             - averageNNZLoad)        / averageNNZLoad) )
            statistics['Nflops_jacobian'].append( (100.0 * (data['Nflops_jacobian'] - averageFLOPSJacobLoad) / averageFLOPSJacobLoad) )
        
        return statistics
    
    def _partitionDAESystem(self, blockIDs, directory):
        # This function returns interprocess_sync_map and equationBlockIndexes
        
        interprocess_sync_map = {}
        equationBlockIndexes  = []
        
        Npe        = self.Npe
        map_oi_bi  = self.analyzer.runtimeInformation['IndexMappings']        
        Nequations = self.analyzer.runtimeInformation['NumberOfEquations']
        
        # Collect all equations and find block indexes in them.
        weights_Nnz          = []         
        weights_Ncs          = []
        weights_Nflops       = []
        weights_Nflops_jacob = []
        for equation in self.analyzer.runtimeInformation['Equations']:
            for eeinfo in equation['EquationExecutionInfos']:
                ncs_items, nflops = eeinfo['ComputeStackInfo'] # tuple: CS.size, CS.flops
                overall_indexes   = eeinfo['VariableIndexes']
                
                nnz           = len(overall_indexes)
                block_indexes = []
                for oi in overall_indexes:
                    bi = map_oi_bi.get(oi, None)
                    # if bi is None then it is not in the map (it is a DOF)
                    if bi != None:
                        block_indexes.append(bi)
                
                # Adjacency data (for METIS)
                equationBlockIndexes.append(block_indexes)
                
                # Vertex weights (for METIS)
                weights_Nnz.append(nnz)
                weights_Ncs.append(ncs_items)
                weights_Nflops.append(nflops)
                weights_Nflops_jacob.append(nnz * nflops)
        
        # All vertex weights together (for internal use)
        equation_weights = weights_Ncs + weights_Nflops + weights_Nnz + weights_Nflops_jacob
        
        if len(self.balancingConstraints) == 0:
            vweights = None
        else:
            vweights = []
            for objective in self.balancingConstraints:
                if objective == 'Ncs':
                    vweights += weights_Ncs
                elif objective == 'Nflops':
                    vweights += weights_Nflops
                elif objective == 'Nnz':
                    vweights += weights_Nnz
                elif objective == 'Nflops_jacobian':
                    vweights += weights_Nflops_jacob
                else:
                    raise RuntimeError('Invalid partitioning objective: %s' % objective)
        
        # Index sets for all processing elements (PE)
        loads_oi        = numpy.zeros((4, Npe)) # Ncs, Nflops, Nnz, Nflops_jacobian
        all_oi          = [set() for i in range(Npe)]
        owned_oi        = [set() for i in range(Npe)]
        foreign_oi      = [set() for i in range(Npe)]
        send_to_oi      = [{}    for i in range(Npe)]
        receive_from_oi = [{}    for i in range(Npe)]
        bi_to_bi_local  = [{}    for i in range(Npe)]
        
        # Perform the partitioning.
        import pymetis
        n_cuts, partitions = pymetis.part_graph(Npe, adjacency = equationBlockIndexes, vweights = vweights)
        #self.logf.write('n_cuts (metis) = %d\n' % n_cuts)
        
        # Generate a list with OWNED block indexes for each PE.
        for ei,pe in enumerate(partitions):
            owned_oi[pe].add(ei)
            loads_oi[0][pe] += equation_weights[0*Nequations + ei] # Ncs
            loads_oi[1][pe] += equation_weights[1*Nequations + ei] # Flops
            loads_oi[2][pe] += equation_weights[2*Nequations + ei] # NNZ
            loads_oi[3][pe] += equation_weights[3*Nequations + ei] # Flops Jacobian
        
        # Generate a list with ALL block indexes for each PE.
        for pe, owned_indexes in enumerate(owned_oi):
            for ei in owned_indexes:
                all_oi[pe].update(equationBlockIndexes[ei])
        
        # Generate a list with FOREIGN block indexes for each PE.
        for pe in range(Npe):
            owned_indexes = owned_oi[pe]
            all_indexes   = all_oi[pe]
            diff = all_indexes.difference(owned_indexes)
            foreign_oi[pe].update(diff)
        
        # Generate dictionaries with block indexes for data exchange (receive_from and send_to) for each PE.
        for pe in range(Npe):
            foreign_indexes = foreign_oi[pe]
            for pe_other in range(Npe):
                owned_indexes_other = owned_oi[pe_other]
                rf = foreign_indexes.intersection(owned_indexes_other)
                if len(rf) > 0:
                    indexes = sorted(rf)
                    receive_from_oi[pe][pe_other] = indexes
                    send_to_oi[pe_other][pe]      = indexes
        
        # Sort owned and foreign indexes.
        for pe in range(Npe):
            owned_oi[pe]   = sorted(owned_oi[pe])
            foreign_oi[pe] = sorted(foreign_oi[pe])
            
        # Create bi_to_bi_local dictionary which maps global block indexes to local block indexes for each PE.
        # The map includes ALL block indexes (OWNED and FOREIGN).
        for pe in range(Npe):
            owned_indexes   = owned_oi[pe]
            foreign_indexes = foreign_oi[pe]
            bi2bi_local     = bi_to_bi_local[pe]
            
            for i,bi in enumerate(owned_indexes):
                bi2bi_local[bi] = i
            
            n = len(owned_indexes)
            for i,bi in enumerate(foreign_indexes):
                bi2bi_local[bi] = n + i
        

        Ncs_average         = numpy.average(loads_oi[0])
        Flops_average       = numpy.average(loads_oi[1])
        NNZ_average         = numpy.average(loads_oi[2])
        Flops_jacob_average = numpy.average(loads_oi[3])
        self.logf.write('Balancing constraints:   %s\n'     % (', '.join(self.balancingConstraints)))
        self.logf.write('Ncs loads:               %s\n'     % loads_oi[0].tolist())
        self.logf.write('Ncs average:             %.1f\n'   % Ncs_average)
        self.logf.write('Flops loads:             %s\n'     % loads_oi[1].tolist())      
        self.logf.write('Flops average:           %.1f\n'   % Flops_average)
        self.logf.write('NNZ loads:               %s\n'     % loads_oi[2].tolist())
        self.logf.write('NNZ average:             %.1f\n'   % NNZ_average)
        self.logf.write('Flops jacobian loads:    %s\n'     % loads_oi[3].tolist())      
        self.logf.write('Flops jacobian average:  %.1f\n'   % Flops_jacob_average)
        
        Ncs_dev          = numpy.abs(loads_oi[0]-Ncs_average)         * 100 / Ncs_average
        Flops_dev        = numpy.abs(loads_oi[1]-Flops_average)       * 100 / Flops_average
        NNZ_dev          = numpy.abs(loads_oi[2]-NNZ_average)         * 100 / NNZ_average
        Flops_jacob_dev  = numpy.abs(loads_oi[3]-Flops_jacob_average) * 100 / Flops_jacob_average
        self.logf.write('Ncs dev %%          = %s\n' % Ncs_dev.tolist())
        self.logf.write('Flops dev %%        = %s\n' % Flops_dev.tolist())            
        self.logf.write('NNZ dev %%          = %s\n' % NNZ_dev.tolist())            
        self.logf.write('Flops Jacob dev %%  = %s\n' % Flops_jacob_dev.tolist())            
        # Populate interprocess_sync_map and write some info.
        for pe in range(Npe):
            owned_indexes   = owned_oi[pe]
            foreign_indexes = foreign_oi[pe]
            receive_from    = receive_from_oi[pe]
            send_to         = send_to_oi[pe]
            bi2bi_local     = bi_to_bi_local[pe]
            Ncs_load        = loads_oi[0][pe]
            Flops_load      = loads_oi[1][pe]
            NNZ_load        = loads_oi[2][pe]
            Flops_Jacob_Load= loads_oi[3][pe]
            interprocess_sync_map[pe] = { 'Nequations'      : len(owned_indexes),
                                          'owned_indexes'   : owned_indexes,
                                          'foreign_indexes' : foreign_indexes,
                                          'bi_to_bi_local'  : bi2bi_local,
                                          'send_to'         : send_to,
                                          'receive_from'    : receive_from,
                                          'Ncs_load'        : Ncs_load,
                                          'FLOPS_load'      : Flops_load,
                                          'NNZ_load'        : NNZ_load,
                                          'FLOPS_Jacob_Load': Flops_Jacob_Load,
                                         }
            
            self.logf.write('\n\nPE = %d (owned: %d; foreign: %d)\n' % (pe, len(owned_indexes), len(foreign_indexes)))
            self.logf.write('  Ncs:          %d\n' % Ncs_load)
            self.logf.write('  Nflops:       %d\n' % Flops_load)
            self.logf.write('  Nnz:          %d\n' % NNZ_load)
            self.logf.write('  Nflops_jacob: %d\n' % Flops_Jacob_Load)
            self.logf.write('  owned_indexes:\n')
            self.logf.write('    %s\n' % owned_indexes)
            self.logf.write('  foreign_indexes:\n')
            self.logf.write('    %s\n' % foreign_indexes)
            self.logf.write('  bi_to_bi_local:\n')
            self.logf.write('    %s\n' % bi2bi_local)
            self.logf.write('  receive_from:\n')
            for pe,rf in receive_from.items():
                self.logf.write('    %d: %s\n' % (pe,rf))
            self.logf.write('  send_to:\n')
            for pe,st in send_to.items():
                self.logf.write('    %d: %s\n' % (pe,st))
        
        #################################################################################
        # Important!
        # Test if the partitioning produce meaningful results:
        #   sometimes the result are equations that depend only on foreign indexes.
        #################################################################################
        # First collect equations for each PE.
        eqn_partitions = [list() for i in range(Npe)]
        for ei,pe in enumerate(partitions):
            eqn = equationBlockIndexes[ei]
            eqn_partitions[pe].append(eqn)
        # Remove foreign indexes and check if the list with local block indexes is empty.
        singular = {}
        for pe in range(Npe):
            bi2bi_local      = bi_to_bi_local[pe]
            equations        = eqn_partitions[pe]
            Nequations_local = len(equations)
            for ei, equation in enumerate(equations):
                equation_with_no_foreign_indexes = []
                for bi in equation:
                    bi_local = bi2bi_local[bi]
                    if bi_local < Nequations_local:
                        equation_with_no_foreign_indexes.append(bi_local)
                if len(equation_with_no_foreign_indexes) == 0:
                    if not pe in singular:
                        singular[pe] = []
                    singular[pe].append(ei)
        if len(singular) > 0:
            msg  = 'After partitioning some equations depend only on foreign indexes.\n'
            msg += 'Consequently, the resulting DAE subsystems are not posed well (their Jacobian matrix is singular).\n'
            msg += 'See %s file for detais.' % self.logf.name
            err = ''
            for pe, equations in singular.items():
                err += '  In partition %d equations %s\n' % (pe, equations)
            self.logf.write('\n\n')
            self.logf.write('%s\n' % msg)
            self.logf.write('%s\n' % err)
            self.errors.append(msg)

            """
            # Perform bandwidth reduction using Cuthill McKee algorithm.
            # This works but the results are indexes scattered around.  
            # Therefore, leave the partitions as they are!
            owned_indexes = owned_oi[pe]
            A  = []
            IA = []
            JA = []
            counter = 0
            IA.append(counter)
            for ei, equation in enumerate(equations):
                equation_with_no_foreign_indexes = []
                for bi in equation:
                    bi_local = bi2bi_local[bi]
                    if bi_local < Nequations_local:
                        equation_with_no_foreign_indexes.append(bi_local)
                if len(equation_with_no_foreign_indexes) == 0:
                    raise RuntimeError('Equation %d in PE %d after partitioning depends only on foreign indexes' % (ei, pe))
                JA.extend(equation_with_no_foreign_indexes)
                counter += len(equation_with_no_foreign_indexes)
                IA.append(counter)
            NNZ = len(JA)
            A = [255] * NNZ
            import scipy.sparse, scipy.sparse.csgraph
            A_csr = scipy.sparse.csr_matrix((A, JA, IA), shape=(Nequations_local,Nequations_local), dtype=numpy.int32)
            perm = scipy.sparse.csgraph.reverse_cuthill_mckee(A_csr, False)
            self.logf.write('PE %d Cuthill McKee' % pe)
            self.logf.write(NNZ)
            # Permute rows and columns (this method is claimed to be faster)
            A_csr.indices = perm.take(A_csr.indices)
            A_csr = A_csr.tocsc()
            A_csr.indices = perm.take(A_csr.indices)
            A_csr = A_csr.tocsr()            
            A_csr_perm = A_csr.toarray()
            #A_csr_perm = A_csr_perm[perm, perm]
            self.logf.write(A_csr_perm.shape)
            # Now update owned_indexes and bi_to_bi_local dictionary
            import png
            filename = os.path.join('/home/ciroki/mpi-exchange', 'incidence_matrix-%05d.png' % pe)
            f = open(filename, 'wb')
            w = png.Writer(Nequations_local, Nequations_local, greyscale=True)
            w.write(f, A_csr_perm.tolist())
            f.close()
            """
            
        diff_as_algebraic_in_other_pe = {}
        for pe in range(Npe):
            foreign_indexes = numpy.array(foreign_oi[pe], dtype=numpy.int32)
            foreign_ids     = blockIDs[foreign_indexes]
            diff_foreign    = set(foreign_indexes[ numpy.where(foreign_ids == cnDifferential)[0] ]) # foreign ids that are diff
            for pe_other in range(Npe):
                owned_indexes_other   = numpy.array(owned_oi[pe_other])
                owned_ids_other       = blockIDs[owned_indexes_other]
                algebraic_owned_other = set(owned_indexes_other[ numpy.where(owned_ids_other == cnAlgebraic)[0] ])
                
                rf = diff_foreign.intersection(algebraic_owned_other)
                if len(rf) > 0: # there are diff foreign vars that are algebraic in the owned PE 
                    if not pe in diff_as_algebraic_in_other_pe:
                        diff_as_algebraic_in_other_pe[pe] = []
                    diff_as_algebraic_in_other_pe[pe].append( (pe_other,rf) )
            
        if len(diff_as_algebraic_in_other_pe) > 0:
            err = 'After partitioning there are differential foreign variables that are algebraic in the owned PE.\n'
            for pe, (pe_other, var_indexes) in diff_as_algebraic_in_other_pe.items():
                err += '  In partition %d variables %s\n' % (pe, var_indexes)
            self.errors.append(err)
    
        return interprocess_sync_map, equationBlockIndexes
        
        
        """ Working printout!!!
        for pe, n_data in mpi_sync_map.items():
            print('[Node %d]:' % pe)
            print('  index_range: [%d - %d)' % (n_data['i_start'], n_data['i_end']))
            print('  all_indexes:')
            print('    %s' % sorted(n_data['all_indexes']))
            print('  owned_indexes:')
            print('    %s' % n_data['owned_indexes'])
            print('  foreign_indexes:')
            print('    %s' % n_data['foreign_indexes'])
            print('  send_to:')
            for sti, st_data in n_data['send_to'].items():
                print('    %d: %s' % (sti, st_data))
            print('  receive_from:')
            for rfi, rf_data in n_data['receive_from'].items():
                print('    %d: %s' % (rfi, rf_data))
        """
