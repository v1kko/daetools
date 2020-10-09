include(../dae.pri)
QT -= core gui
TARGET = cdaeCore
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${OPEN_CS_INCLUDE}

LIBS += $${SOLIBS_RPATH_SL}

win32::QMAKE_CXXFLAGS += -DMODEL_EXPORTS

LIBS +=	$${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${DAE_CAPE_THERMO_PACKAGE_LIB} \
        $${BOOST_LIBS}

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
            finite_element_model.cpp \
            global_functions.cpp \
            if.cpp \
            io_impl.cpp \
            model_port_arrays.cpp \
            model.cpp \
            nodes.cpp \
            nodes_array.cpp \
            nodes_compute_stack.cpp \
            cs_export.cpp \
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
            logs.h \
            inlines_varparam_array.h \
            nodes_array.h \
            tcpiplog.h \
            optimization.h \
            export.h \
            units_io.h \
            simplify_node.h \
            thermo_package.h \
            call_stats.h \
            compute_stack_kernels_openmp.h

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

include(../dae_install_library.pri)
QMAKE_CXXFLAGS += -fpermissive
