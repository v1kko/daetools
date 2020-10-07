TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = Config \
    CoolPropThermoPackage \
    Units \
    Core \
    DataReporting \
    IDAS_DAESolver \
    Activity \
    pyCore \
    pyActivity \
    pyDataReporting \
    pyIDAS \
    pyUnits \
    simulation_loader \
    fmi \
    fmi_ws

HEADERS += config.h \
           daetools.h \
           daetools-pymod.h \
           daetools-core.h \
           variable_types.h
