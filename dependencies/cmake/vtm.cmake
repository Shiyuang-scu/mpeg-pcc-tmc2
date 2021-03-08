cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
SET( VTM_VERSION         VTM-11.2 )
SET( VTM_DIR             ${CMAKE_SOURCE_DIR}/dependencies/${VTM_VERSION} )
set( VTM_LIB_SOURCE_DIR  ${VTM_DIR}/source/Lib )
MESSAGE("Clone and build VTM libraries: ${VTM_LIB_SOURCE_DIR}") 

IF( NOT EXISTS "${VTM_DIR}/CMakeLists.txt" )
  MESSAGE("VTM clone: ${VTM_LIB_SOURCE_DIR}")
   EXECUTE_PROCESS( COMMAND git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git/ ${VTM_DIR} RESULT_VARIABLE ret)   
   IF( NOT ${ret} EQUAL "0")
     MESSAGE( FATAL_ERROR "Error during the VTM git clone process. Check that git is well installed on your system.")
   ENDIF()
   EXECUTE_PROCESS( COMMAND git checkout ${VTM_VERSION} WORKING_DIRECTORY ${VTM_DIR} RESULT_VARIABLE ret)   
ELSE()
  MESSAGE("VTM already cloned: ${VTM_LIB_SOURCE_DIR}")
ENDIF()

IF( NOT EXISTS "${VTM_DIR}/PATCHED" )
  MESSAGE("VTM patch: ${VTM_DIR}")
  SET( VTM_PATCH ${CMAKE_SOURCE_DIR}/dependencies/vtm-modification/adaptions_for_vtm_11_2.patch )
  EXECUTE_PROCESS( COMMAND git apply ${VTM_PATCH} --whitespace=nowarn WORKING_DIRECTORY ${VTM_DIR} RESULT_VARIABLE ret )
  IF( NOT ${ret} EQUAL "0")
    MESSAGE( FATAL_ERROR "Error during the VTM patch process. Check that git is well installed on your system." )
  ENDIF()
  FILE( WRITE ${VTM_DIR}/PATCHED "VTM patched with: " ${VTM_PATCH} )   
ELSE()
  MESSAGE("VTM already patched: ${VTM_DIR}")
ENDIF()

function(add_vtm_library module)
  file(GLOB cppSourceFiles "${VTM_LIB_SOURCE_DIR}/${module}/*.cpp")
  file(GLOB cSourceFiles "${VTM_LIB_SOURCE_DIR}/${module}/*.c")
  file(GLOB headerFiles "${VTM_LIB_SOURCE_DIR}/${module}/*.h")
  add_library(${module}_vtm ${cppSourceFiles} ${cSourceFiles} ${headerFiles})
  target_compile_definitions( ${module}_vtm PUBLIC )
  set_property(TARGET ${module}_vtm PROPERTY CXX_CLANG_TIDY) # no clang-tidy
  add_library(VPCC::${module}_vtm ALIAS ${module}_vtm)
endfunction()

add_vtm_library(libmd5)
target_compile_features(libmd5 PUBLIC cxx_std_11)
target_include_directories(libmd5 PUBLIC "$<BUILD_INTERFACE:${VTM_LIB_SOURCE_DIR}>")

file(GLOB cppSourceFiles "${VTM_LIB_SOURCE_DIR}/CommonLib/*.cpp")
file(GLOB cSourceFiles "${VTM_LIB_SOURCE_DIR}/CommonLib/*.c")
file(GLOB headerFiles "${VTM_LIB_SOURCE_DIR}/CommonLib/*.h")
file(GLOB X86_SRC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/*.cpp" )
file(GLOB X86_INC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/*.h" )
file(GLOB AVX_SRC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/avx/*.cpp" )
file(GLOB AVX2_SRC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/avx2/*.cpp" )
file(GLOB SSE42_SRC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/sse42/*.cpp" )
file(GLOB SSE41_SRC_FILES "${VTM_LIB_SOURCE_DIR}/CommonLib/x86/sse41/*.cpp" )
add_library(CommonLib ${cppSourceFiles} ${cSourceFiles} ${headerFiles} ${X86_SRC_FILES} ${X86_INC_FILES} ${AVX_SRC_FILES} ${AVX2_SRC_FILES} ${SSE42_SRC_FILES} ${SSE41_SRC_FILES})
set_property(TARGET CommonLib PROPERTY CXX_CLANG_TIDY) # no clang-tidy
add_library(VPCC::CommonLib ALIAS CommonLib)
target_include_directories( CommonLib PUBLIC ${VTM_LIB_SOURCE_DIR}/CommonLib/. ${VTM_LIB_SOURCE_DIR}/CommonLib/.. ${VTM_LIB_SOURCE_DIR}/CommonLib/x86 )
target_link_libraries(CommonLib PRIVATE libmd5_vtm)
target_compile_features(CommonLib PUBLIC cxx_std_11)
target_include_directories(CommonLib PUBLIC "$<BUILD_INTERFACE:${VTM_LIB_SOURCE_DIR}>")
target_compile_definitions(CommonLib PUBLIC "$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>")
target_compile_options(CommonLib PUBLIC "$<$<CXX_COMPILER_ID:Clang>:-w>")
target_compile_options(CommonLib PUBLIC "$<$<CXX_COMPILER_ID:GNU>:-w>")
# set needed compile definitions
set_property( SOURCE ${SSE41_SRC_FILES} APPEND PROPERTY COMPILE_DEFINITIONS USE_SSE41 )
set_property( SOURCE ${SSE42_SRC_FILES} APPEND PROPERTY COMPILE_DEFINITIONS USE_SSE42 )
set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_DEFINITIONS USE_AVX )
set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_DEFINITIONS USE_AVX2 )
# set needed compile flags
if( MSVC )
  set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_FLAGS "/arch:AVX" )
  set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_FLAGS "/arch:AVX2" )
elseif( UNIX OR MINGW )
  set_property( SOURCE ${SSE41_SRC_FILES} APPEND PROPERTY COMPILE_FLAGS "-msse4.1" )
  set_property( SOURCE ${SSE42_SRC_FILES} APPEND PROPERTY COMPILE_FLAGS "-msse4.2" )
  set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_FLAGS "-mavx" )
  set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_FLAGS "-mavx2" )
endif()


add_vtm_library(Utilities)
target_link_libraries(Utilities_vtm PUBLIC CommonLib)

add_vtm_library(DecoderLib)
target_link_libraries(DecoderLib_vtm PUBLIC CommonLib)

add_vtm_library(EncoderLib)
target_link_libraries(EncoderLib_vtm PUBLIC CommonLib)
