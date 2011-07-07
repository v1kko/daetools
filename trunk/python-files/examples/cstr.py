#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             cstr.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

"""
"""

import sys, math
from daetools.pyDAE import *
from daetools.solvers.daeMinpackLeastSq import *
from time import localtime, strftime
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
       
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Caf   = daeParameter("Ca_f",  eReal, self, "")
        self.Tc    = daeParameter("T_c",   eReal, self, "")
        self.V     = daeParameter("V",     eReal, self, "")
        self.ro    = daeParameter("&rho;", eReal, self, "")
        self.cp    = daeParameter("cp",    eReal, self, "")
        self.R     = daeParameter("R",     eReal, self, "")
        self.UA    = daeParameter("UA",    eReal, self, "")

        # Parameters to estimate
        self.dH    = daeVariable("&Delta;H_r", no_t, self, "")
        self.E     = daeVariable("E",          no_t, self, "")
        self.k0    = daeVariable("k_0",        no_t, self, "")
        
        # Inputs
        self.q     = daeVariable("q",    molar_flowrate_t,      self, "")
        self.Tf    = daeVariable("T_f",  temperature_t,         self, "")
        
        # Measured variables
        self.T  = daeVariable("T",  temperature_t,         self, "")
        self.Ca = daeVariable("Ca", molar_concentration_t, self, "")

    def DeclareEquations(self):
        eq = self.CreateEquation("MolarBalance", "")
        eq.Residual = self.V() * self.Ca.dt() - \
                      self.q() * (self.Caf() - self.Ca()) + \
                      1E10*self.k0() * self.V() * Exp(-(1E4*self.E()) / (self.R() * self.T())) * self.Ca()

        eq = self.CreateEquation("HeatBalance", "")
        eq.Residual = self.ro() * self.cp() * self.V() * self.T.dt() - \
                      self.q()  * self.ro() * self.cp() * (self.Tf() - self.T()) - \
                      self.V()  * 1E4 * self.dH() * 1E10*self.k0() * Exp(-(1E4*self.E())/(self.R() * self.T())) * self.Ca() - \
                      self.UA() * (self.Tc() - self.T())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("cstr")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.Caf.SetValue(1)
        self.m.Tc.SetValue(270)
        self.m.ro.SetValue(1000)
        self.m.cp.SetValue(0.239)
        self.m.V.SetValue(100)
        self.m.R.SetValue(8.31451)
        self.m.UA.SetValue(5e4)

    def SetUpVariables(self):
        # Inputs
        self.m.q.AssignValue(100)
        self.m.Tf.AssignValue(350)
        
        # Parameters 
        self.m.dH.AssignValue(5)         # * 1E4
        self.m.E.AssignValue(7.27519625) # * 1E4
        self.m.k0.AssignValue(7.2)       # * 1E10

        self.m.T.SetInitialCondition(305)
        self.m.Ca.SetInitialCondition(0.5)
    
    def SetUpParameterEstimation(self):
        self.SetMeasuredVariable(self.m.Ca)
        self.SetMeasuredVariable(self.m.T)
        
        self.SetInputVariable(self.m.Tf)
        self.SetInputVariable(self.m.q)
        
        self.SetModelParameter(self.m.dH, 0, 100,  5.0)
        self.SetModelParameter(self.m.E,  0, 100, 10.0)
        self.SetModelParameter(self.m.k0, 0, 100, 10.0)

def plotConfidenceEllipsoids(minpack, x_param_index, y_param_index, confidences, x_label, y_label):
    fig = plt.figure()
    legend = []
    for c in range(0, len(confidences)):
        confidence = confidences[c]
        x_ellipse, y_ellipse, x0, y0 = minpack.getConfidenceEllipsoid(x_param_index = x_param_index, 
                                                                      y_param_index = y_param_index, 
                                                                      confidence    = confidence)
        # print x_ellipse, y_ellipse
        ax = fig.add_subplot(111, aspect='auto')
        ax.plot(x_ellipse, y_ellipse)
        legend.append(str(confidence)+'%')
        if c == len(confidences)-1:
            ax.plot(x0, y0, 'o')
            legend.append('opt')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    ax.legend(legend, frameon=False)
    return fig

def plotExpFitComparison(minpack, measured_variable_index, experiment_index, x_label, y_label, legend):
    fig = plt.figure()
    x_axis, y_exp, y_fit = minpack.getFit_Dyn(measured_variable_index = measured_variable_index, experiment_index = experiment_index)
    ax = fig.add_subplot(111, aspect='auto')
    ax.plot(x_axis, y_fit, 'blue', x_axis, y_exp, 'o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(legend, frameon=False)
    return fig

if __name__ == "__main__":
    try:
        log          = daePythonStdOutLog()
        daesolver    = daeIDAS()
        datareporter = daeTCPIPDataReporter()
        simulation   = simTutorial()
        minpack      = daeMinpackLeastSq()

        # Enable reporting of all variables
        simulation.m.SetReportingOn(True)

        # Set the time horizon and the reporting interval
        simulation.ReportingInterval = 0.5
        simulation.TimeHorizon = 10

        # Connect data reporter
        simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        if(datareporter.Connect("", simName) == False):
            sys.exit()

        # Some info about the experiments
        Nparameters          = 3
        Ninput_variables     = 2
        Nmeasured_variables  = 2
        Nexperiments         = 9
        Ntime_points         = 20

        # Experimental data:
        data = [
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [280.0, 80.0],
                    [[0.66329477068663312, 0.77389744837948538, 0.8481214449170339, 0.89787413185859655, 0.9312080838961081, 0.95354406021042792, 0.96850625010695612, 0.9785397843853928, 0.98526544152724405, 0.9897754556341305, 0.99279782271498829, 0.99482101468444217, 0.99617133278550096, 0.99708394607766349, 0.99769513183793856, 0.99810334451652194, 0.99837924337436668, 0.99856377625066517, 0.99868754012247363, 0.9987702868903916],
                    [280.5258339015146, 274.64444413974894, 273.25080207191127, 272.92344082713657, 272.84781851276909, 272.83023478557669, 272.82675303414965, 272.82626076765257, 272.82649133152222, 272.82678765502089, 272.82702458225901, 272.82719268422636, 272.82730755956248, 272.82738490852006, 272.82743709723439, 272.82747206484066, 272.82749571243505, 272.82751153650935, 272.82752215178169, 272.82752924930736]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [300.0, 80.0],
                    [[0.66293263785426393, 0.77336066258060043, 0.84748560895204039, 0.89716972370653802, 0.93046156610292374, 0.95275824176607826, 0.96769291581711159, 0.97769672957502263, 0.98439862836646996, 0.98889324806142065, 0.99189721310954126, 0.99390926945864444, 0.99525619501672302, 0.99615853921462527, 0.99676045614524433, 0.99716308467195891, 0.99743214901830402, 0.99761399765173653, 0.99773612252360322, 0.99781761527112411],
                    [284.80454109970503, 279.92055987276336, 278.76088243805606, 278.48783810593005, 278.42490728761607, 278.41277082562027, 278.41143527272015, 278.41224656452448, 278.41311146589555, 278.41377657653902, 278.41426089292435, 278.41458673708547, 278.41480705884788, 278.41495731262171, 278.41505522677983, 278.41512151561096, 278.41516606984823, 278.41519600993576, 278.41521616042559, 278.41522959537286]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [320.0, 80.0],
                    [[0.66244945472517291, 0.77254796863213404, 0.84645801694874501, 0.89597938874753835, 0.92913822820716929, 0.95133208820170734, 0.96618942935409302, 0.97612674713864622, 0.98278335775001346, 0.98723447571641865, 0.99021406973819504, 0.99220826661646799, 0.99354344427785402, 0.99443697290405941, 0.9950354748532968, 0.9954355975083039, 0.9957027768881842, 0.99588916891169854, 0.9960182854267009, 0.99610031053027359],
                    [289.10150343614964, 285.22694397683114, 284.3037943178104, 284.08952868442992, 284.04398498991128, 284.03827927994752, 284.03946535679989, 284.0422986948027, 284.04402516700873, 284.04539114780539, 284.04630059870959, 284.04692145038194, 284.04733710589574, 284.04761572144082, 284.04780244510681, 284.04792724935783, 284.04801058860869, 284.04806874544687, 284.04810902265939, 284.04813459964942]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [280.0, 100.0],
                    [[0.6953078275735316, 0.81479395840732916, 0.887339380445617, 0.93134315964828907, 0.95802661024767355, 0.9742013107886941, 0.98400534787519178, 0.98994959495389512, 0.99355048622276754, 0.99573224786698955, 0.99705877413372679, 0.99785867252875693, 0.99834561404109612, 0.99864018072317695, 0.99882212584190033, 0.99893361617777576, 0.99900055884302374, 0.99904050261104826, 0.99906711358246136, 0.99908343661533316],
                    [280.16051044548561, 274.76607226272324, 273.60398932423925, 273.35588716939185, 273.30468846040333, 273.29536524063661, 273.2941711577779, 273.29404220581631, 273.29433740252233, 273.29447387356953, 273.2945384026192, 273.29459386138245, 273.29464587102899, 273.29467449072649, 273.29469026643096, 273.29470038412683, 273.2947063992828, 273.29470999665364, 273.2947124621885, 273.29471389875687]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [300.0, 100.0],
                    [[0.69487526267654098, 0.81416943278416731, 0.88661006174141765, 0.93053748464542041, 0.95716324513878681, 0.97330095440153763, 0.98307209475053758, 0.98899510107881616, 0.99258191567147702, 0.99475770560937304, 0.99607664533331552, 0.9968766491079889, 0.99736227256934729, 0.99765597735685907, 0.99783417920756867, 0.99794147138082445, 0.99800738270259559, 0.99804716040931385, 0.99807412851336985, 0.99809082988940245],
                    [285.30980814256543, 280.99668364948513, 280.06941930599237, 279.87370721209055, 279.83435974358895, 279.82792552044589, 279.82792608189226, 279.82859165551133, 279.82917571876209, 279.82956810387981, 279.82981710750067, 279.82997035521288, 279.83006422202266, 279.83012096307152, 279.83015543194369, 279.83017619902068, 279.83018896498169, 279.83019666940902, 279.83020189154303, 279.8302051234698]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [320.0, 100.0],
                    [[0.69425683968538887, 0.8131217087253888, 0.88528723959551681, 0.92902001898460607, 0.95550260356082306, 0.97153501282117216, 0.98124125642731719, 0.98711564853484324, 0.99066940918511359, 0.99282196754610874, 0.99412257292693706, 0.99491130352989265, 0.99538675663165832, 0.99567326272878021, 0.99584917461415012, 0.995954540780842, 0.99602331479743333, 0.99606622359877106, 0.99608812567548843, 0.9961001739044224],
                    [290.48607063212688, 287.28062245668366, 286.59189919148071, 286.45034249343763, 286.42611462689626, 286.42521061831911, 286.42736121691684, 286.42940657083022, 286.4307822278725, 286.43163482380425, 286.4321551649295, 286.43247479756258, 286.43266792502175, 286.43278446845017, 286.43285603254856, 286.4328989104888, 286.43292690756385, 286.43294435736345, 286.43295327743192, 286.43295818691689]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [280.0, 120.0],
                    [[0.72427534167530339, 0.84828790540955834, 0.91640383449097751, 0.9537853149401252, 0.97429383352458965, 0.98554633342910847, 0.99171828149149788, 0.99510415646219086, 0.99696088992534693, 0.9979801327383987, 0.99853899622055775, 0.99884551934454524, 0.99901329557356378, 0.99910468713295963, 0.99915329163121647, 0.9991840087768763, 0.99920366209263767, 0.9992095359162193, 0.99921192945298265, 0.99921256024635341],
                    [279.83792200385659, 274.89113758422616, 273.93084198801404, 273.74591470053412, 273.71158299491719, 273.70494216421241, 273.70458566473502, 273.70449938051269, 273.70466748532903, 273.70486824310188, 273.70496358486452, 273.70502677485558, 273.70494778839003, 273.70489439367117, 273.70486819647618, 273.70487717413744, 273.70490121801561, 273.70487180359271, 273.7048721662805, 273.70488303409508]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [300.0, 120.0],
                    [[0.72377750643775962, 0.84757969553219747, 0.91559035146516277, 0.95290190449910872, 0.97336119404660915, 0.98457821173044302, 0.99072816261931684, 0.99409824487052467, 0.99594796109118811, 0.99696101988836372, 0.99751673914035166, 0.99782149070805781, 0.99798809033645897, 0.99808062554281096, 0.99812575893395328, 0.99815124895345897, 0.99816742768370981, 0.99817879816766619, 0.99818527043048055, 0.99818781735808182],
                    [285.78961413931142, 281.98216060375688, 281.24140068703963, 281.10012077300809, 281.07540178092069, 281.07188355469583, 281.07205025628667, 281.07265800393122, 281.07288418511945, 281.07313489304511, 281.07323198633924, 281.07331957014122, 281.07335861429704, 281.07337023572563, 281.07336793406421, 281.07341613167586, 281.07337699075572, 281.07339004422823, 281.07340574216812, 281.0734074218376]]
                ), 
                (
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                    [320.0, 120.0],
                    [[0.72302279082102217, 0.84629264826226902, 0.91397877813315298, 0.9510737139486628, 0.97139066184831613, 0.98250791092434131, 0.98859874300120198, 0.99193345267719302, 0.99375801313227285, 0.99475787839900764, 0.99530758813771958, 0.99560564485329317, 0.99576860148596569, 0.9958534709929282, 0.99589985400251513, 0.99592649225760055, 0.99594141277577675, 0.99594939105404645, 0.99595744382373286, 0.99595944091686517],
                    [291.78287106402041, 289.14118989048865, 288.62943101437173, 288.53832351214282, 288.52634746533334, 288.52791100360446, 288.53003115373127, 288.53164428395849, 288.53248388428568, 288.53294785250796, 288.53325844514876, 288.53341426221999, 288.53349554297131, 288.53354115552833, 288.53356528282802, 288.53357905342779, 288.53358674212075, 288.53359084771887, 288.53359499165862, 288.53359601651465]]
                )
               ]
       
        # Initialize MinpackLeastSq
        minpack.Initialize(simulation, 
                           daesolver, 
                           datareporter, 
                           log, 
                           experimental_data            = data,
                           print_residuals_and_jacobian = False,
                           enforce_parameters_bounds    = False,
                           minpack_leastsq_arguments    = {'ftol'   : 1E-8, 
                                                           'xtol'   : 1E-8, 
                                                           'factor' : 0.1} )
        
        # Save the model report and the runtime model report
        simulation.m.SaveModelReport(simulation.m.Name + ".xml")
        simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

        # Run
        minpack.Run()

        # Print the results
        print 'Status:', minpack.msg
        print 'Number of function evaluations =', minpack.infodict['nfev']
        print 'Estimated parameters\' values:', minpack.p_estimated
        print 'The real parameters\' values:', ['5.0', '7.27519625', '7.2']
        print 'χ2 =', minpack.x2
        print 'Standard deviation =', minpack.sigma
        print 'Covariance matrix:'
        print minpack.cov_x
        
        plotConfidenceEllipsoids(minpack, 0, 1, [90,95,99], 'dHr', 'E')
        plotConfidenceEllipsoids(minpack, 0, 2, [90,95,99], 'dHr', 'k0')
        plotConfidenceEllipsoids(minpack, 1, 2, [90,95,99], 'E',   'k0')
        
        plotExpFitComparison(minpack, 0, 5, 'x', 'y', ['Ca-fit', 'Ca-exp'])
        plotExpFitComparison(minpack, 1, 5, 'x', 'y', ['T-fit', 'T-exp'])
        
        plt.show()

    except Exception, e:
        print str(e)
        
    finally:
        minpack.Finalize()
