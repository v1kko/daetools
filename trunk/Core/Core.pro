include(../dae.pri)
QT -= core \
    gui
TARGET = Core
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR}
SOURCES += xmlreadstream.cpp \
    xmlfunctions.cpp \
    xmlfile.cpp \
    variablearray.cpp \
    variable_types.cpp \
    variable.cpp \
    stn.cpp \
    if.cpp \
    state_transition.cpp \
    state.cpp \
    stdafx.cpp \
    ports.cpp \
    partial.cpp \
    parameter.cpp \
    object.cpp \
    nodes.cpp \
    model_port_arrays.cpp \
    model.cpp \
    helpers.cpp \
    fpu_commands.cpp \
    equation.cpp \
    domain.cpp \
    dllmain.cpp \
    distributed_variable_array.cpp \
    condnode.cpp \
    condition.cpp \
    class_factory.cpp \
    bool_array.cpp \
    block.cpp \
    adouble.cpp \
    setup_nodes.cpp \
    xmltag.cpp \
    io_impl.cpp \
    adouble_array.cpp \
    nodes_array.cpp \
    setup_node_array.cpp \
    base_logging.cpp \
    stdafx.cpp \
    dllmain.cpp \
    global_functions.cpp \
    tcpiplog.cpp \
    objfun_constraints.cpp
HEADERS += xmlfunctions.h \
    xmlfile.h \
    stdafx.h \
    solver.h \
    nodes.h \
    macros.h \
    log.h \
    io_impl.h \
    io_helpers.h \
    io_fns.h \
    io.h \
    inlines_portarray.h \
    inlines_modelarray.h \
    inlines_io.h \
    inlines_equation.h \
    inlines_varparam_templates.h \
    helpers.h \
    definitions.h \
    datareporting.h \
    coreimpl.h \
    core.h \
    class_factory.h \
    adouble.h \
    activity.h \
    base_logging.h \
    inlines_varparam_array.h \
    nodes_array.h \
    tcpiplog.h \
    optimization.h
