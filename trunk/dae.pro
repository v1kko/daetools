TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
	Units \
	pyCore \
	pyActivity \
	pyDataReporting \
	pyIDAS \
	pyUnits
 
HEADERS += dae.h \
    dae_develop.h
