include(../dae.pri)
QT -= gui
TARGET = Examples
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHONDIR}
win32 { 
    win32-msvc2008:INCLUDEPATH += ../sundials-2.3.0/include \
        ../sundials-2.3.0/src
    win32-msvc2008:LIBS += $${DAE_DEST_DIR}/Core$${DAE_DEBUG}.lib \
        $${DAE_DEST_DIR}/Simulation$${DAE_DEBUG}.lib \
        $${DAE_DEST_DIR}/DataReporters$${DAE_DEBUG}.lib \
        $${DAE_DEST_DIR}/Solver$${DAE_DEBUG}.lib
    win32-g++:INCLUDEPATH += ../sundials-2.3.0/include \
        ../sundials-2.3.0/src
    win32-g++:LIBS += -L$${DAE_DEST_DIR} \
        -lCore$${DAE_DEBUG} \
        -lSimulation$${DAE_DEBUG} \
        -lDataReporters$${DAE_DEBUG} \
        -lSolver$${DAE_DEBUG}
}
else:INCLUDEPATH += ../sundials-2.3.0/include \
    ../sundials-2.3.0/src
unix:LIBS += -L$${DAE_DEST_DIR} \
    -lCore$${DAE_DEBUG} \
    -lSimulation$${DAE_DEBUG} \
    -lDataReporters$${DAE_DEBUG} \
    -lSolver$${DAE_DEBUG} \
    -lboost_python
SOURCES += examples.cpp \
    stdafx.h \
    dllmain.cpp
HEADERS += variable_types.h \
    stdafx.h \
    test_models.h \
    straight_fin1.h \
    heat_cond1.h \
    first.h \
    cstr.h \
    conduction.h \
    py_example.h \
    template_model.h
