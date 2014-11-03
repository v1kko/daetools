include(../dae.pri)

TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

QMAKE_CXXFLAGS += $$system($${PYTHON}-config --cflags)
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                ../release

LIBS += -lcdaeSimulationLoader
