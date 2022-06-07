# -*- mode: cmake; -*-
# vi: set ft=cmake:

#
# CMakeLists.txt
# RVO2 Library
#
# Copyright 2008 University of North Carolina at Chapel Hill
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please send all bug reports to <geom@cs.unc.edu>.
#
# The authors may be contacted via:
#
# Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
# Dept. of Computer Science
# 201 S. Columbia St.
# Frederick P. Brooks, Jr. Computer Science Bldg.
# Chapel Hill, N.C. 27599-3175
# United States of America
#
# <https://gamma.cs.unc.edu/RVO2/>
#

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(RVO)

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/modules")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # cmake-lint: disable=C0301
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build; options are Debug Release RelWithDebInfo MinSizeRel"
    FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
    STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

include(CTest)

if(WIN32)
  set(BUILD_SHARED_LIBS OFF)
else()
  option(BUILD_SHARED_LIBS "Build all libraries as shared" ON)
endif()

include(GNUInstallDirs)

set(RVO_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(RVO_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
set(RVO_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})
set(RVO_LIBRARY RVO)
SET(GL_LIBRARY /usr/lib/x86_64-linux-gnu/libGL.so.1)

set(RVO_NAME "RVO2 Library")
set(RVO_HOMEPAGE_URL https://gamma.cs.unc.edu/RVO2/)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_OPENMP "Enable OpenMP if available" ON)
set(ENABLE_OPENMP OFF)

if(ENABLE_OPENMP)

  find_package(OpenMP MODULE)
endif()

option(ENABLE_INTERPROCEDURAL_OPTIMIZATION
  "Enable interprocedural optimization if supported" OFF)

if(ENABLE_INTERPROCEDURAL_OPTIMIZATION)
  include(CheckIPOSupported)

  check_ipo_supported(RESULT RVO_INTERPROCEDURAL_OPTIMIZATION_SUPPORTED
    LANGUAGES CXX)

  if(RVO_INTERPROCEDURAL_OPTIMIZATION_SUPPORTED)
    message(STATUS "Interprocedural optimization is supported")
  else()
    message(STATUS "Interprocedural optimization is NOT supported")
  endif()
else()
  set(RVO_INTERPROCEDURAL_OPTIMIZATION_SUPPORTED)
endif()

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include(CTest)

add_subdirectory(src)

include(CPack)
