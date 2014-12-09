include(../dae.pri)

TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

QMAKE_CXXFLAGS += $$system($${PYTHON}-config --cflags)
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        -lcdaeSimulationLoader \
        -ldaetools_fmi_cs-Linux_x86_64

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                ../release

