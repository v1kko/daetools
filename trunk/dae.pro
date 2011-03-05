TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
    BONMIN_MINLPSolver \
	#NLOPT_NLPSolver \
	pyCore \
	pyActivity \
	pyDataReporting \
	pyIDAS \
	pyBONMIN \
	ExternalFunction
 
HEADERS += dae.h \
    dae_develop.h
