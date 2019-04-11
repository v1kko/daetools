cmake_minimum_required(VERSION 3.0.0)

project(OpenCS)

# Find include files
file(GLOB HEADERS_OPENCS        OpenCS/*.h)
file(GLOB HEADERS_EVALUATORS    OpenCS/evaluators/*.h)
file(GLOB HEADERS_MODELS        OpenCS/models/*.h)
file(GLOB HEADERS_SIMULATORS    OpenCS/simulators/*.h)

# Find source files
file(GLOB SOURCES_EVALUATORS    OpenCS/evaluators/*.cpp)
file(GLOB SOURCES_OPENMP        OpenCS/openmp/*.cpp)
file(GLOB SOURCES_OPENCL        OpenCS/opencl/*.cpp)
file(GLOB SOURCES_MODELS        OpenCS/models/*.cpp)
file(GLOB SOURCES_SIMULATORS    OpenCS/simulators/*.cpp)

set(SOURCES_CS_SIMULATOR        OpenCS/simulators/cs_simulator_binary.cpp)

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

set(CVODES_DIR    "${PROJECT_SOURCE_DIR}/cvodes"        CACHE PATH "Path to Sundials CVodes root directory")
set(IDAS_DIR      "${PROJECT_SOURCE_DIR}/idas"          CACHE PATH "Path to Sundials IDAS root directory")
set(Trilinos_DIR  "${PROJECT_SOURCE_DIR}/trilinos"      CACHE PATH "Path to Trilinos root directory")
set(Boost_DIR     "${PROJECT_SOURCE_DIR}/boost"         CACHE PATH "Path to boost root directory")
set(METIS_DIR     "${PROJECT_SOURCE_DIR}/metis"         CACHE PATH "Path to METIS root directory")
set(CLAPACK_DIR   "${PROJECT_SOURCE_DIR}/clapack"       CACHE PATH "Path to CLapack library directory")
set(HDF5_DIR      "${PROJECT_SOURCE_DIR}/hdf5/install"  CACHE PATH "Path to HDF5 library directory")

if(APPLE)
  find_package(OpenCL REQUIRED)
elseif(UNIX)
  # /usr/local/cuda/include 
  set(OpenCL_INCLUDE_DIRS "/opt/intel/opencl/include"     CACHE STRING "Path to OpenCL include directory")
  set(OpenCL_LIBRARIES    "-L/opt/intel/opencl -lOpenCL"  CACHE STRING "OpenCL libraries")
elseif(MSVC)
  find_package(OpenCL REQUIRED)
endif()

set(Trilinos_INCLUDE_DIRS "${Trilinos_DIR}/build/include")
set(Trilinos_LIBRARY_DIRS "${Trilinos_DIR}/build/lib")

set(METIS_INCLUDE_DIR    "${METIS_DIR}/build/include")
set(METIS_LIB_DIR        "${METIS_DIR}/build/lib")

find_package(MPI REQUIRED)

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
include_directories(${HDF5_DIR}/include)

if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "")
  link_directories("${MPI_C_LINK_FLAGS}")
endif()
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${CVODES_DIR}/build/lib)
link_directories(${IDAS_DIR}/build/lib)
link_directories(${METIS_LIB_DIR})
link_directories(${HDF5_DIR}/lib)

if(WIN32)
  link_directories(${CLAPACK_DIR}/build/lib)
endif(WIN32)

if(UNIX)
  set(STD_FILESYSTEM stdc++fs)
  set(METIS_LIB            "-lmetis")
  set(OpenCS_CVODES_LIBS   "-lsundials_cvodes -lsundials_nvecparallel")
  set(OpenCS_IDAS_LIBS     "-lsundials_idas -lsundials_nvecparallel")
  set(OpenCS_TRILINOS_LIBS "-lml -lifpack -laztecoo -lamesos -lepetraext -ltriutils -lepetra -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -llapack -lblas -lgfortran -lm")
  set(HDF5_LIBS            "${HDF5_DIR}/lib/libhdf5.a"
                           "${HDF5_DIR}/lib/libszip.a" 
                           "${HDF5_DIR}/lib/libz.a"
                           "-lm -ldl")
endif(UNIX)
if(MSVC)
  set(STD_FILESYSTEM "")
  set(METIS_LIB            "metis.lib")
  set(OpenCS_CVODES_LIBS   "sundials_cvodes.lib sundials_nvecparallel.lib")
  set(OpenCS_IDAS_LIBS     "sundials_idas.lib sundials_nvecparallel.lib")
  set(OpenCS_TRILINOS_LIBS "ml.lib ifpack.lib aztecoo.lib amesos.lib epetraext.lib triutils.lib epetra.lib teuchosremainder.lib teuchosnumerics.lib teuchoscomm.lib teuchosparameterlist.lib teuchoscore.lib lapack.lib blas.lib libf2c.lib")
  set(HDF5_LIBS            "${HDF5_DIR}/lib/libhdf5.lib"
                           "${HDF5_DIR}/lib/libszip.lib" 
                           "${HDF5_DIR}/lib/libzlib.lib")
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
#target_link_libraries(OpenCS_Models OpenCS_Evaluators)
#target_link_libraries(OpenCS_Models "${OpenCL_LIBRARIES}")
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
target_link_libraries(OpenCS_Simulators "${MPI_CXX_LIBRARIES}")
target_link_libraries(OpenCS_Simulators "${HDF5_LIBS}")

#target_link_libraries(OpenCS_Simulators "-L${PROJECT_SOURCE_DIR}/../release -lcdaeTrilinos_LASolver")
#message("MPI_C_INCLUDE_PATH = ${MPI_C_INCLUDE_PATH}")
#message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
#message("MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
#message("MPI_C_LINK_FLAGS = ${MPI_C_LINK_FLAGS}")

if(APPLE)
    # use, i.e. don't skip the full RPATH for the build tree
    set(CMAKE_SKIP_BUILD_RPATH  FALSE)

    # when building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    set(CMAKE_INSTALL_RPATH "@loader_path/../lib") 

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

    # the RPATH to be used when installing, but only if it's not a system directory
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "@loader_path/../lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
      set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
    endif("${isSystemDir}" STREQUAL "-1")

elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN" "$ORIGIN/../lib")
  
elseif(MSVC)
  target_compile_definitions(OpenCS_Evaluators PUBLIC OpenCS_EVALUATORS_EXPORTS)
  target_compile_definitions(OpenCS_Models     PUBLIC OpenCS_MODELS_EXPORTS)
  target_compile_definitions(OpenCS_Simulators PUBLIC OpenCS_SIMULATORS_EXPORTS)
endif()

if(${HDF5_FOUND})
  target_compile_definitions(OpenCS_Simulators PUBLIC OpenCS_HDF5_Enabled)
  #message("OpenCS_HDF5_Enabled = ${HDF5_FOUND}")
endif()

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

add_executable(csSimulator ${SOURCES_CS_SIMULATOR})
add_dependencies(csSimulator OpenCS_Evaluators OpenCS_Models OpenCS_Simulators)

include_directories(csSimulator OpenCS/simulators)

target_link_libraries(csSimulator OpenCS_Simulators)
target_link_libraries(csSimulator OpenCS_Models)
target_link_libraries(csSimulator OpenCS_Evaluators)
target_link_libraries(csSimulator "${MPI_CXX_LIBRARIES}")
target_link_libraries(csSimulator "${MPI_C_LIBRARIES}")

# Not required anymore on GNU/Linux, since the rpath is set to "$ORIGIN:$ORIGIN/../lib"
if(MSVC)
  install(TARGETS OpenCS_Evaluators   DESTINATION bin)
  install(TARGETS OpenCS_Models       DESTINATION bin)
  install(TARGETS OpenCS_Simulators   DESTINATION bin)
endif(MSVC)
install(TARGETS csSimulator           DESTINATION bin)

###############################################################################
#                           Examples
###############################################################################
project(examples)

include_directories(${Boost_DIR})

foreach(example dae_example_1 dae_example_2 dae_example_3
                ode_example_1 ode_example_2 ode_example_3)
  
  set(SOURCES_EXAMPLE_${example} examples/${example}.cpp)
  add_executable(${example} ${SOURCES_EXAMPLE_${example}})
  
  add_dependencies(${example} OpenCS_Evaluators 
                              OpenCS_Models 
                              OpenCS_Simulators)
  
  target_link_libraries(${example} OpenCS_Simulators 
                                   OpenCS_Models 
                                   OpenCS_Evaluators
                                   "${MPI_CXX_LIBRARIES}" 
                                   "${MPI_C_LIBRARIES}")
  
  # Not required anymore on GNU/Linux, since the rpath is set to "$ORIGIN:$ORIGIN/../lib"
  if(MSVC)
    install(TARGETS OpenCS_Evaluators   DESTINATION examples)
    install(TARGETS OpenCS_Models       DESTINATION examples)
    install(TARGETS OpenCS_Simulators   DESTINATION examples)
  endif(MSVC)

  install(TARGETS ${example} DESTINATION examples)
endforeach(example)
