# Module for locating libnuma
#
# Read-only variables:
#   NUMA_FOUND
#     Indicates that the library has been found.
#
#   NUMA_INCLUDE_DIRS
#     Points to the libnuma include directory.
#
#   NUMA_LIBRARY_DIR
#     Points to the directory that contains the libraries.
#     The content of this variable can be passed to link_directories.
#
#   NUMA_LIBRARY
#     Points to the libnuma that can be passed to target_link_libararies.
#
# Copyright (c) 2015 Steve Borho

include(FindPackageHandleStandardArgs)

find_path(NUMA_INCLUDE_DIRS
  NAMES numa.h
  HINTS ${NUMA_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "NUMA include directory")

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${NUMA_ROOT_DIR}
  DOC "NUMA library")

if (NUMA_LIBRARY)
    get_filename_component(NUMA_LIBRARY_DIR ${NUMA_LIBRARY} PATH)
endif()

mark_as_advanced(NUMA_INCLUDE_DIRS NUMA_LIBRARY_DIR NUMA_LIBRARY)

find_package_handle_standard_args(Numa REQUIRED_VARS NUMA_INCLUDE_DIRS NUMA_LIBRARY)

if(NUMA_FOUND)
    if(NOT TARGET Numa::Numa)
        add_library(Numa::Numa SHARED IMPORTED)
    endif()
    if(NUMA_INCLUDE_DIRS)
        set_target_properties(Numa::Numa PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${NUMA_LIBRARY}")
        set_target_properties(Numa::Numa PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${NUMA_LIBRARY}")
    endif()
endif()

