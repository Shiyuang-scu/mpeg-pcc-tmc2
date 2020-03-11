


file( GLOB_RECURSE ALL_CXX_SOURCE_FILES source/*.cpp source/*.h )

# clang-format 
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND clang-format
    -i
    -style=file
    ${ALL_CXX_SOURCE_FILES}
    )
endif()

# clang-tidy
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
#   get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#   SET( INCLUDE_ARGS "" )
#   foreach(dir ${dirs})
#     SET( INCLUDE_ARGS "${INCLUDE_ARGS} -I${dir}" )
#     message( ${INCLUDE_ARGS})
#   endforeach()
   
    set( INCLUDE_ARGS " -I${CMAKE_CURRENT_SOURCE_DIR}/source/dependencies/nanoflann \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/dependencies/libmd5  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/dependencies/tbb/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/dependencies/program-options-lite  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibBitstreamCommon/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibBitstreamReader/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibBitstreamWriter/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibCommon/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibDecoder/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibEncoder/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/lib/PccLibMetrics/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/app/PccAppDecoder/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/app/PccAppEncoder/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/app/PccAppMetrics/include   \
    -I${CMAKE_CURRENT_SOURCE_DIR}/source/app/PccAppParser/include  " )
     
    message( " clang-tidy ${ALL_CXX_SOURCE_FILES} --  -system-headers=0 ${INCLUDE_ARGS} " )

    # -std=c++14 -std=c++11
    add_custom_target(
      clang-tidy
      COMMAND clang-tidy
      -p ./build/  
      -header-filter=include/*.h
      -system-headers=0
      -fix 
      -fix-errors 
      ${ALL_CXX_SOURCE_FILES}
      --
      ${INCLUDE_ARGS}
      )
endif()