# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main

# Include any dependencies generated for this target.
include examples/CMakeFiles/Roadmap.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/Roadmap.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/Roadmap.dir/flags.make

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o: examples/CMakeFiles/Roadmap.dir/flags.make
examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o: examples/Roadmap.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Roadmap.dir/Roadmap.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples/Roadmap.cpp

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Roadmap.dir/Roadmap.cpp.i"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples/Roadmap.cpp > CMakeFiles/Roadmap.dir/Roadmap.cpp.i

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Roadmap.dir/Roadmap.cpp.s"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples/Roadmap.cpp -o CMakeFiles/Roadmap.dir/Roadmap.cpp.s

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.requires:

.PHONY : examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.requires

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.provides: examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/Roadmap.dir/build.make examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.provides.build
.PHONY : examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.provides

examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.provides.build: examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o


# Object files for target Roadmap
Roadmap_OBJECTS = \
"CMakeFiles/Roadmap.dir/Roadmap.cpp.o"

# External object files for target Roadmap
Roadmap_EXTERNAL_OBJECTS =

examples/Roadmap: examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o
examples/Roadmap: examples/CMakeFiles/Roadmap.dir/build.make
examples/Roadmap: src/libRVO.so.2.0.3
examples/Roadmap: examples/CMakeFiles/Roadmap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Roadmap"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Roadmap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/Roadmap.dir/build: examples/Roadmap

.PHONY : examples/CMakeFiles/Roadmap.dir/build

examples/CMakeFiles/Roadmap.dir/requires: examples/CMakeFiles/Roadmap.dir/Roadmap.cpp.o.requires

.PHONY : examples/CMakeFiles/Roadmap.dir/requires

examples/CMakeFiles/Roadmap.dir/clean:
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples && $(CMAKE_COMMAND) -P CMakeFiles/Roadmap.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/Roadmap.dir/clean

examples/CMakeFiles/Roadmap.dir/depend:
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples/CMakeFiles/Roadmap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/Roadmap.dir/depend

