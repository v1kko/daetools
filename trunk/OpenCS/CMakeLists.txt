cmake_minimum_required(VERSION 3.0.0)

project(OpenCS)

# Find include files
file(GLOB HEADERS_OPENCS        OpenCS/*.h)
file(GLOB HEADERS_EVALUATORS    OpenCS/evaluators/*.h)
file(GLOB HEADERS_MODELS        OpenCS/models/*.h)
set(HEADERS_SIMULATORS          OpenCS/simulators/daesimulator.h
                                OpenCS/simulators/cs_simulators.h)

# Find source files
file(GLOB SOURCES_EVALUATORS    OpenCS/evaluators/*.cpp)
file(GLOB SOURCES_OPENMP        OpenCS/openmp/*.cpp)
file(GLOB SOURCES_OPENCL        OpenCS/opencl/*.cpp)
file(GLOB SOURCES_MODELS        OpenCS/models/*.cpp)
set(SOURCES_SIMULATORS          OpenCS/simulators/auxiliary.cpp 
                                OpenCS/simulators/config.cpp 
                                OpenCS/simulators/daesolver.cpp
                                OpenCS/simulators/odeisolver.cpp
                                OpenCS/simulators/daemodel.cpp 
                                OpenCS/simulators/daesimulation.cpp
                                OpenCS/simulators/preconditioner_jacobi.cpp 
                                OpenCS/simulators/preconditioner_ifpack.cpp 
                                OpenCS/simulators/preconditioner_ml.cpp
                                OpenCS/simulators/cs_simulators.cpp)

set(SOURCES_ODE_SIMULATOR       OpenCS/simulators/cs_ode_simulator.cpp)
set(SOURCES_DAE_SIMULATOR       OpenCS/simulators/cs_dae_simulator.cpp)

if(UNIX)
  set(CMAKE_C_FLAGS             "-std=c99 -fopenmp -fPIC")
  set(CMAKE_CXX_FLAGS           "-std=c++11 -fopenmp -fPIC")
  set(CMAKE_SHARED_LINKER_FLAGS "-fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS    "-fopenmp")
endif(UNIX)

if(MSVC)
  set(CMAKE_C_FLAGS             "/std:c99 /openmp")
  set(CMAKE_CXX_FLAGS           "/std:c++14 /openmp /EHsc")
  set(CMAKE_SHARED_LINKER_FLAGS "")
  set(CMAKE_EXE_LINKER_FLAGS    "")
endif(MSVC)

# Required for recording overheads during a simulation
#add_definitions(-DSTEP_DURATIONS)

set(CVODES_DIR           "${PROJECT_SOURCE_DIR}/cvodes"     CACHE PATH "Path to Sundials CVodes root directory")
set(IDAS_DIR             "${PROJECT_SOURCE_DIR}/idas"       CACHE PATH "Path to Sundials IDAS root directory")
set(Trilinos_DIR         "${PROJECT_SOURCE_DIR}/trilinos"   CACHE PATH "Path to Trilinos root directory")
set(Boost_DIR            "${PROJECT_SOURCE_DIR}/boost"      CACHE PATH "Path to boost root directory")
set(METIS_DIR            "${PROJECT_SOURCE_DIR}/metis"      CACHE PATH "Path to METIS root directory")
set(CLAPACK_DIR          "${PROJECT_SOURCE_DIR}/clapack"    CACHE PATH "Path to CLapack library directory")

if(UNIX)
  # /usr/local/cuda/include 
  set(OpenCL_INCLUDE_DIRS "/opt/intel/opencl/include"     CACHE STRING "Path to OpenCL include directory")
  set(OpenCL_LIBRARIES    "-L/opt/intel/opencl -lOpenCL"  CACHE STRING "OpenCL libraries")
endif(UNIX)
if(MSVC)
  find_package(OpenCL REQUIRED)
endif(MSVC)

set(Trilinos_INCLUDE_DIRS "${Trilinos_DIR}/build/include")
set(Trilinos_LIBRARY_DIRS "${Trilinos_DIR}/build/lib")

set(METIS_INCLUDE_DIR    "${METIS_DIR}/build/include")
set(METIS_LIB_DIR        "${METIS_DIR}/build/lib")

find_package(MPI REQUIRED)
#find_library(METIS_LIB   metis  PATHS "${METIS_LIB_DIR}")

###############################################################################
#                           OpenCS
###############################################################################
# Include header files
include_directories(.)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${CVODES_DIR} ${CVODES_DIR}/build/include)
include_directories(SYSTEM ${IDAS_DIR}   ${IDAS_DIR}/build/include)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Boost_DIR})
include_directories(${METIS_INCLUDE_DIR})

if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "")
  link_directories("${MPI_C_LINK_FLAGS}")
endif()
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${CVODES_DIR}/build/lib)
link_directories(${IDAS_DIR}/build/lib)
link_directories(${CLAPACK_DIR}/build/lib)
link_directories(${METIS_LIB_DIR})

if(UNIX)
  set(STD_FILESYSTEM stdc++fs)
  set(METIS_LIB            "-lmetis")
  set(OpenCS_CVODES_LIBS   "-lsundials_cvodes -lsundials_nvecparallel")
  set(OpenCS_IDAS_LIBS     "-lsundials_idas -lsundials_nvecparallel")
  set(OpenCS_TRILINOS_LIBS "-lml -lifpack -laztecoo -lamesos -lepetraext -ltriutils -lepetra -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -llapack -lblas -lgfortran -lm")
endif(UNIX)
if(MSVC)
  set(STD_FILESYSTEM "")
  set(METIS_LIB            "metis.lib")
  set(OpenCS_CVODES_LIBS   "sundials_cvodes.lib sundials_nvecparallel.lib")
  set(OpenCS_IDAS_LIBS     "sundials_idas.lib sundials_nvecparallel.lib")
  set(OpenCS_TRILINOS_LIBS "ml.lib ifpack.lib aztecoo.lib amesos.lib epetraext.lib triutils.lib epetra.lib teuchosremainder.lib teuchosnumerics.lib teuchoscomm.lib teuchosparameterlist.lib teuchoscore.lib lapack.lib blas.lib libf2c.lib")
endif(MSVC)

# Create libraries
add_library(OpenCS_Evaluators SHARED ${SOURCES_EVALUATORS})
add_library(OpenCS_Models     SHARED ${SOURCES_MODELS})
add_library(OpenCS_Simulators SHARED ${SOURCES_SIMULATORS})
add_dependencies(OpenCS_Models OpenCS_Evaluators)
add_dependencies(OpenCS_Simulators OpenCS_Evaluators OpenCS_Models)

# OpenCS depends only on OpenMP and OpenCL libs.
target_link_libraries(OpenCS_Evaluators "${OpenCL_LIBRARIES}")

# OpenCS_Models depends on OpenMP, OpenCL, and MPI libs.
target_link_libraries(OpenCS_Models OpenCS_Evaluators)
target_link_libraries(OpenCS_Models "${OpenCL_LIBRARIES}")
target_link_libraries(OpenCS_Models "${STD_FILESYSTEM}")
target_link_libraries(OpenCS_Models "${MPI_C_LIBRARIES}")
if(METIS_LIB)
  target_link_libraries(OpenCS_Models "${METIS_LIB}")
endif()

target_link_libraries(OpenCS_Simulators OpenCS_Evaluators OpenCS_Models)
target_link_libraries(OpenCS_Simulators "${OpenCS_TRILINOS_LIBS}")
target_link_libraries(OpenCS_Simulators "${OpenCL_LIBRARIES}")
target_link_libraries(OpenCS_Simulators "${OpenCS_CVODES_LIBS}")
target_link_libraries(OpenCS_Simulators "${OpenCS_IDAS_LIBS}")
target_link_libraries(OpenCS_Simulators "${STD_FILESYSTEM}")
target_link_libraries(OpenCS_Simulators "${MPI_C_LIBRARIES}")

#target_link_libraries(OpenCS_Simulators "-L${PROJECT_SOURCE_DIR}/../release -lcdaeTrilinos_LASolver")
#message("${PROJECT_SOURCE_DIR}/../release")

set(CMAKE_INSTALL_RPATH "$ORIGIN" "$ORIGIN/../lib")

# Install libraries
install(TARGETS OpenCS_Evaluators   DESTINATION lib)
install(TARGETS OpenCS_Models       DESTINATION lib)
install(TARGETS OpenCS_Simulators   DESTINATION lib)

# Install library headers
install(FILES ${HEADERS_OPENCS}                      DESTINATION include/OpenCS)
install(FILES ${HEADERS_EVALUATORS}                  DESTINATION include/OpenCS/evaluators)
install(FILES ${HEADERS_MODELS}                      DESTINATION include/OpenCS/models)
install(FILES ${HEADERS_SIMULATORS}                  DESTINATION include/OpenCS/simulators)

###############################################################################
#               Reference implementation ODE/DAE simulators
###############################################################################
project(OpenCS_Simulators)

add_executable(csSimulator_ODE ${SOURCES_ODE_SIMULATOR})
add_executable(csSimulator_DAE ${SOURCES_DAE_SIMULATOR})
add_dependencies(csSimulator_ODE OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(csSimulator_DAE OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)

include_directories(csSimulator_ODE OpenCS/simulators)
include_directories(csSimulator_DAE OpenCS/simulators)

target_link_libraries(csSimulator_ODE OpenCS_Simulators)
target_link_libraries(csSimulator_ODE OpenCS_Models)
target_link_libraries(csSimulator_ODE OpenCS_Evaluators)
target_link_libraries(csSimulator_ODE "${MPI_CXX_LIBRARIES}")
target_link_libraries(csSimulator_ODE "${MPI_C_LIBRARIES}")

target_link_libraries(csSimulator_DAE OpenCS_Simulators)
target_link_libraries(csSimulator_DAE OpenCS_Models)
target_link_libraries(csSimulator_DAE OpenCS_Evaluators)
target_link_libraries(csSimulator_DAE "${MPI_CXX_LIBRARIES}")
target_link_libraries(csSimulator_DAE "${MPI_C_LIBRARIES}")

# Not required anymore on GNU.Linux, since the rpath is set to "$ORIGIN:$ORIGIN/../lib"
if(MSVC)
  install(TARGETS OpenCS_Evaluators   DESTINATION bin)
  install(TARGETS OpenCS_Models       DESTINATION bin)
  install(TARGETS OpenCS_Simulators   DESTINATION bin)
endif(MSVC)
install(TARGETS csSimulator_ODE      DESTINATION bin)
install(TARGETS csSimulator_DAE      DESTINATION bin)

###############################################################################
#                           Examples
###############################################################################
project(examples)

set(SOURCES_DAE_EXAMPLE_1 examples/dae_example_1.cpp)
set(SOURCES_DAE_EXAMPLE_2 examples/dae_example_2.cpp)
set(SOURCES_DAE_EXAMPLE_3 examples/dae_example_3.cpp)
set(SOURCES_ODE_EXAMPLE_1 examples/ode_example_1.cpp)
set(SOURCES_ODE_EXAMPLE_2 examples/ode_example_2.cpp)
set(SOURCES_ODE_EXAMPLE_3 examples/ode_example_3.cpp)

add_executable(dae_example_1 ${SOURCES_DAE_EXAMPLE_1})
add_executable(dae_example_2 ${SOURCES_DAE_EXAMPLE_2})
add_executable(dae_example_3 ${SOURCES_DAE_EXAMPLE_3})
add_executable(ode_example_1 ${SOURCES_ODE_EXAMPLE_1})
add_executable(ode_example_2 ${SOURCES_ODE_EXAMPLE_2})
add_executable(ode_example_3 ${SOURCES_ODE_EXAMPLE_3})

add_dependencies(dae_example_1 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(dae_example_2 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(dae_example_3 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(ode_example_1 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(ode_example_2 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)
add_dependencies(ode_example_3 OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)

include_directories(${Boost_DIR})

target_link_libraries(dae_example_1 OpenCS_Simulators)
target_link_libraries(dae_example_1 OpenCS_Models)
target_link_libraries(dae_example_1 OpenCS_Evaluators)
target_link_libraries(dae_example_1 "${MPI_CXX_LIBRARIES}")
target_link_libraries(dae_example_1 "${MPI_C_LIBRARIES}")

target_link_libraries(dae_example_2 OpenCS_Simulators)
target_link_libraries(dae_example_2 OpenCS_Models)
target_link_libraries(dae_example_2 OpenCS_Evaluators)
target_link_libraries(dae_example_2 "${MPI_CXX_LIBRARIES}")
target_link_libraries(dae_example_2 "${MPI_C_LIBRARIES}")

target_link_libraries(dae_example_3 OpenCS_Simulators)
target_link_libraries(dae_example_3 OpenCS_Models)
target_link_libraries(dae_example_3 OpenCS_Evaluators)
target_link_libraries(dae_example_3 "${MPI_CXX_LIBRARIES}")
target_link_libraries(dae_example_3 "${MPI_C_LIBRARIES}")

target_link_libraries(ode_example_1 OpenCS_Simulators)
target_link_libraries(ode_example_1 OpenCS_Models)
target_link_libraries(ode_example_1 OpenCS_Evaluators)
target_link_libraries(ode_example_1 "${MPI_CXX_LIBRARIES}")
target_link_libraries(ode_example_1 "${MPI_C_LIBRARIES}")

target_link_libraries(ode_example_2 OpenCS_Simulators)
target_link_libraries(ode_example_2 OpenCS_Models)
target_link_libraries(ode_example_2 OpenCS_Evaluators)
target_link_libraries(ode_example_2 "${MPI_CXX_LIBRARIES}")
target_link_libraries(ode_example_2 "${MPI_C_LIBRARIES}")

target_link_libraries(ode_example_3 OpenCS_Simulators)
target_link_libraries(ode_example_3 OpenCS_Models)
target_link_libraries(ode_example_3 OpenCS_Evaluators)
target_link_libraries(ode_example_3 "${MPI_CXX_LIBRARIES}")
target_link_libraries(ode_example_3 "${MPI_C_LIBRARIES}")

# Not required anymore on GNU.Linux, since the rpath is set to "$ORIGIN:$ORIGIN/../lib"
if(MSVC)
  install(TARGETS OpenCS_Evaluators   DESTINATION examples)
  install(TARGETS OpenCS_Models       DESTINATION examples)
  install(TARGETS OpenCS_Simulators   DESTINATION examples)
endif(MSVC)

install(TARGETS dae_example_1 DESTINATION examples)
install(TARGETS dae_example_2 DESTINATION examples)
install(TARGETS dae_example_3 DESTINATION examples)

install(TARGETS ode_example_1 DESTINATION examples)
install(TARGETS ode_example_2 DESTINATION examples)
install(TARGETS ode_example_3 DESTINATION examples)
