include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeCore
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH += $${BOOSTDIR} \
	           $${MPI_INCLUDE}

LIBS +=	$${MPI_LIBS}

SOURCES += xmlreadstream.cpp \
    xmlfunctions.cpp \
    xmlfile.cpp \
    variablearray.cpp \
    variable_types.cpp \
    variable.cpp \
    stn.cpp \
    if.cpp \
    state.cpp \
    stdafx.cpp \
    ports.cpp \
	event_port.cpp \
	action.cpp \
	on_event_actions.cpp \
	on_condition_actions.cpp \
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
    objfun_constraints.cpp \
    external_function.cpp \
    event_handling.cpp

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
    optimization.h \
    export.h \
    units_io.h \ 
    event_handling.h

#win32{
#QMAKE_POST_LINK = $${COPY_FILES} $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = $${COPY_FILES} lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} ../dae.h         $${HEADERS_DIR})
#INSTALL_HEADERS = $$system($${COPY_FILES} ../dae_develop.h $${HEADERS_DIR})
#INSTALL_HEADERS = $$system($${COPY_FILES} ../config.h      $${HEADERS_DIR})
#INSTALL_HEADERS = $$system($${COPY_FILES} ../nlp_common.h  $${HEADERS_DIR})
#INSTALL_HEADERS = $$system($${COPY_FILES} dae.pri          $${HEADERS_DIR})

#INSTALL_HEADERS = $$system($${COPY_FILES} definitions.h    $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} core.h           $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} log.h            $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} activity.h       $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} optimization.h   $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} solver.h         $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} datareporting.h  $${HEADERS_DIR}/Core)

#INSTALL_HEADERS = $$system($${COPY_FILES} coreimpl.h       $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} helpers.h        $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} class_factory.h  $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} base_logging.h   $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} adouble.h        $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} export.h         $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} io.h             $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} io_impl.h        $${HEADERS_DIR}/Core)
#INSTALL_HEADERS = $$system($${COPY_FILES} xmlfile.h        $${HEADERS_DIR}/Core)

#######################################################
#                Install files
#######################################################
root_headers.path  = $${HEADERS_DIR}
root_headers.files = ../dae.h  \        
                     ../dae_develop.h  \
                     ../config.h  \     
                     ../nlp_common.h  \ 
                     ../dae.pri         

core_headers.path  = $${HEADERS_DIR}/Core
core_headers.files = definitions.h  \
                     core.h  \         
                     log.h  \          
                     activity.h  \     
                     optimization.h  \ 
                     solver.h  \       
                     datareporting.h  \
                     coreimpl.h  \     
                     helpers.h  \      
                     class_factory.h  \
                     base_logging.h  \ 
                     adouble.h  \      
                     export.h  \       
                     io.h  \           
                     io_impl.h  \      
                     xmlfile.h      

core_libs.path         = $${STATIC_LIBS_DIR}
win32::core_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::core_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += root_headers core_headers core_libs

