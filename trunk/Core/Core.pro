include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeCore
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH += $${BOOSTDIR} \
               $${MPI_INCLUDE}

LIBS +=	$${DAE_CONFIG_LIB} \
        $${MPI_LIBS}

SOURCES +=  action.cpp \
            adouble.cpp \
            adouble_array.cpp \
            base_logging.cpp \
            block.cpp \
            compute_stack_kernels_openmp.cpp \
            condnode.cpp \
            condition.cpp \
            class_factory.cpp \
            domain.cpp \
            distributed_variable_array.cpp \
            dllmain.cpp \
            equation.cpp \
            event_port.cpp \
            external_function.cpp \
            event_handling.cpp \
            finite_element_model.cpp \
            global_functions.cpp \
            if.cpp \
            io_impl.cpp \
            model_port_arrays.cpp \
            model.cpp \
            nodes.cpp \
            nodes_array.cpp \
            nodes_compute_stack.cpp \
            object.cpp \
            objfun_constraints.cpp \
            ode_model.cpp \
            on_event_actions.cpp \
            on_condition_actions.cpp \
            ports.cpp \
            partial.cpp \
            parameter.cpp \
            setup_nodes.cpp \
            stn.cpp \
            state.cpp \
            setup_node_array.cpp \
            stdafx.cpp \
            tcpiplog.cpp \
            thermo_package.cpp \
            variablearray.cpp \
            variable_types.cpp \
            variable.cpp \
            xmltag.cpp \
            xmlreadstream.cpp \
            xmlfunctions.cpp \
            xmlfile.cpp

HEADERS +=  xmlfunctions.h \
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
            event_handling.h \
            simplify_node.h \
            thermo_package.h \
            compute_stack.h \
            compute_stack_kernels_openmp.h

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

