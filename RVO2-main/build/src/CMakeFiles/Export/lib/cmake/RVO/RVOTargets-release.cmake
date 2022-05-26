#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "RVO::RVO" for configuration "Release"
set_property(TARGET RVO::RVO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(RVO::RVO PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libRVO.so.2.0.3"
  IMPORTED_SONAME_RELEASE "libRVO.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS RVO::RVO )
list(APPEND _IMPORT_CHECK_FILES_FOR_RVO::RVO "${_IMPORT_PREFIX}/lib/libRVO.so.2.0.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
