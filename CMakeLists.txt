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
#     http://www.apache.org/licenses/LICENSE-2.0
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
# <http://gamma.cs.unc.edu/RVO2/>
#

cmake_minimum_required(VERSION 2.8)
project(RVO)

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  MESSAGE(STATUS "Build type: Debug")
  IF(MSVC)
    SET(CMAKE_CXX_FLAGS "/openmp /bigobj -Wall -std=c++14 ${CMAKE_CXX_FLAGS}")
  ELSE(MSVC)
    SET(CMAKE_CXX_FLAGS "-fopenmp -O0 -Wall -fPIC -std=c++14 -Wno-unused-function -Wno-unused-local-typedefs ${CMAKE_CXX_FLAGS}")
  ENDIF(MSVC)
ELSE()
  MESSAGE(STATUS "Build type: Release")
  IF(MSVC)
    SET(CMAKE_CXX_FLAGS "/openmp /bigobj -Wall -std=c++14 ${CMAKE_CXX_FLAGS}")
  ELSE(MSVC)
    SET(CMAKE_CXX_FLAGS "-fopenmp -O3 -Wall -fPIC -std=c++14 -Wno-unused-function -Wno-unused-local-typedefs ${CMAKE_CXX_FLAGS}")
  ENDIF(MSVC)
ENDIF()
#SET(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
#set(CMAKE_BUILD_TYPE “Release”)
#set(CMAKE_CXX_FLAGS " -fopenmp -fPIC -std=c++14  -O3")
message(STATUS "Using compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(CTest)

add_subdirectory(src)
add_subdirectory(examples)

include(CPack)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
