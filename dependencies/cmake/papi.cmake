cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

IF( WIN32 OR MSVC OR MSYS OR MINGW OR APPLE )
  SET( ENABLE_PAPI_PROFILING OFF )
  MESSAGE( "PAPI profiling forced disable for this operating system" )
ENDIF()

IF( ENABLE_PAPI_PROFILING )
  MESSAGE( "PAPI profiling enable (ENABLE_PAPI_PROFILING = " ${ENABLE_PAPI_PROFILING} " ).")
  IF( NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/dependencies/papi/" )    
    EXECUTE_PROCESS( COMMAND git clone https://bitbucket.org/icl/papi.git ${CMAKE_SOURCE_DIR}/dependencies/papi/; )    
  ENDIF()
  IF ( NOT EXISTS ${CMAKE_SOURCE_DIR}/dependencies/papi/src/libpapi.a )
    EXECUTE_PROCESS( COMMAND ./configure WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/papi/src/ )
    EXECUTE_PROCESS( COMMAND make -j4    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/papi/src/ )
  ENDIF()
  SET( CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -I${CMAKE_SOURCE_DIR}/dependencies/papi/src/ " )
  SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_SOURCE_DIR}/dependencies/papi/src/libpapi.a ")
ENDIF()
