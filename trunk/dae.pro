TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
	#NLOPT_NLPSolver \
	pyCore \
	pyActivity \
	pyDataReporting \
	pyIDAS \
	ExternalFunction
 
HEADERS += dae.h \
    dae_develop.h
