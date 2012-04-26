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
 
HEADERS += config.h \
           dae_develop.h \
           variable_types.h \
           dae.h
