TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
	pyCore \
	pyActivity \
	pyDataReporting \
	pyIDAS
	#ExternalFunction
 
HEADERS += dae.h \
    dae_develop.h
