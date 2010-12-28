TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
	IPOPT_NLPSolver \
	pyCore \
	pyActivity \
	pyDataReporting \
	pyIDAS \
	pyIPOPT
 
HEADERS += dae.h \
    dae_develop.h
