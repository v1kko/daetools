TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporters \
    Simulation \
    Solver \
#	daepython \
	pyCore \
	pyActivity \
	pyDataReporting \
	pySolver

HEADERS += dae.h \
    dae_develop.h
