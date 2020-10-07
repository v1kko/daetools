cmake_minimum_required(VERSION 3.0.0)

project(daetools_examples)

# Specify the platform directory to be used for headers and libraries.
if(APPLE)
  set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/../Darwin_x86_64")
elseif(UNIX)
  set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/../Linux_x86_64")
elseif(MSVC)
  if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/../Windows_win64")
  else()
    set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/../Windows_win32")
  endif()
endif()
message("PLATFORM_DIR = ${PLATFORM_DIR}")

# Headers (still, boost is required).
include_directories(../include
                    ../../boost)

link_directories(${PLATFORM_DIR}/lib)

# Set install directory (if default is used).
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PLATFORM_DIR}/examples CACHE PATH " " FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if(UNIX)
  set(CMAKE_C_FLAGS             "-std=c99 -fPIC")
  set(CMAKE_CXX_FLAGS           "-std=c++11 -fPIC")

  set(DAE_CONFIG_LIB            -lcdaeConfig)
  set(DAE_CORE_LIB              -lcdaeCore)
  set(DAE_DATAREPORTING_LIB     -lcdaeDataReporting)
  set(DAE_ACTIVITY_LIB          -lcdaeActivity)
  set(DAE_IDAS_SOLVER_LIB       -lcdaeIDAS_DAESolver)
  set(DAE_UNITS_LIB             -lcdaeUnits)
  set(DAE_IPOPT_SOLVER_LIB      -lcdaeIPOPT_NLPSolver)
  set(DAE_SUPERLU_SOLVER_LIB    -lcdaeSuperLU_LASolver)

  set(CMAKE_INSTALL_RPATH "$ORIGIN" "$ORIGIN/../lib")
elseif(MSVC)
  set(CMAKE_C_FLAGS   "/std:c99 ")
  set(CMAKE_CXX_FLAGS "/std:c++14 /EHsc")

  set(DAE_CONFIG_LIB            cdaeConfig.lib)
  set(DAE_CORE_LIB              cdaeCore.lib)
  set(DAE_DATAREPORTING_LIB     cdaeDataReporting.lib)
  set(DAE_ACTIVITY_LIB          cdaeActivity.lib)
  set(DAE_IDAS_SOLVER_LIB       cdaeIDAS_DAESolver.lib)
  set(DAE_UNITS_LIB             cdaeUnits.lib)
  set(DAE_IPOPT_SOLVER_LIB      cdaeIPOPT_NLPSolver.lib)
  set(DAE_SUPERLU_SOLVER_LIB    cdaeSuperLU_LASolver.lib)
endif()

foreach(example whats_the_time
                tutorial1 tutorial2 tutorial4 tutorial5 tutorial6 tutorial14
                opt_tutorial1)

  set(SOURCES_EXAMPLE_${example} ${example}.cpp)
  add_executable(${example} ${SOURCES_EXAMPLE_${example}})

  target_link_libraries(${example} ${DAE_DATAREPORTING_LIB}
                                   ${DAE_CORE_LIB}
                                   ${DAE_IDAS_SOLVER_LIB}
                                   ${DAE_UNITS_LIB}
                                   ${DAE_ACTIVITY_LIB})
  if(${example} STREQUAL "opt_tutorial1")
      target_link_libraries(${example} ${DAE_IPOPT_SOLVER_LIB})
  elseif(${example} STREQUAL "tutorial5")
      target_link_libraries(${example} ${DAE_SUPERLU_SOLVER_LIB})
  endif()

  install(TARGETS ${example} DESTINATION .)
endforeach(example)
