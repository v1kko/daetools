TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Config \
    Core \
    DataReporting \
    Activity \
    IDAS_DAESolver \ 
    Units \
    pyCore \
    pyActivity \
    pyDataReporting \
    pyIDAS \
    pyUnits \
    simulation_loader \
    fmi

HEADERS += config.h \
           dae_develop.h \
           variable_types.h \
           dae.h
