TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporters \
    Simulation \
    Solver \ 
	NLPSolver \
	pyCore \
	pyActivity \
	pyDataReporting \
	pySolver \
	pyIPOPT

HEADERS += dae.h \
    dae_develop.h
