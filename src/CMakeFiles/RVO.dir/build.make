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
include src/CMakeFiles/RVO.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/RVO.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/RVO.dir/flags.make

src/CMakeFiles/RVO.dir/Agent.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/Agent.cpp.o: src/Agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/RVO.dir/Agent.cpp.o"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RVO.dir/Agent.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Agent.cpp

src/CMakeFiles/RVO.dir/Agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/Agent.cpp.i"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Agent.cpp > CMakeFiles/RVO.dir/Agent.cpp.i

src/CMakeFiles/RVO.dir/Agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/Agent.cpp.s"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Agent.cpp -o CMakeFiles/RVO.dir/Agent.cpp.s

src/CMakeFiles/RVO.dir/Agent.cpp.o.requires:

.PHONY : src/CMakeFiles/RVO.dir/Agent.cpp.o.requires

src/CMakeFiles/RVO.dir/Agent.cpp.o.provides: src/CMakeFiles/RVO.dir/Agent.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/RVO.dir/build.make src/CMakeFiles/RVO.dir/Agent.cpp.o.provides.build
.PHONY : src/CMakeFiles/RVO.dir/Agent.cpp.o.provides

src/CMakeFiles/RVO.dir/Agent.cpp.o.provides.build: src/CMakeFiles/RVO.dir/Agent.cpp.o


src/CMakeFiles/RVO.dir/KdTree.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/KdTree.cpp.o: src/KdTree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/RVO.dir/KdTree.cpp.o"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RVO.dir/KdTree.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/KdTree.cpp

src/CMakeFiles/RVO.dir/KdTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/KdTree.cpp.i"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/KdTree.cpp > CMakeFiles/RVO.dir/KdTree.cpp.i

src/CMakeFiles/RVO.dir/KdTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/KdTree.cpp.s"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/KdTree.cpp -o CMakeFiles/RVO.dir/KdTree.cpp.s

src/CMakeFiles/RVO.dir/KdTree.cpp.o.requires:

.PHONY : src/CMakeFiles/RVO.dir/KdTree.cpp.o.requires

src/CMakeFiles/RVO.dir/KdTree.cpp.o.provides: src/CMakeFiles/RVO.dir/KdTree.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/RVO.dir/build.make src/CMakeFiles/RVO.dir/KdTree.cpp.o.provides.build
.PHONY : src/CMakeFiles/RVO.dir/KdTree.cpp.o.provides

src/CMakeFiles/RVO.dir/KdTree.cpp.o.provides.build: src/CMakeFiles/RVO.dir/KdTree.cpp.o


src/CMakeFiles/RVO.dir/Obstacle.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/Obstacle.cpp.o: src/Obstacle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/RVO.dir/Obstacle.cpp.o"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RVO.dir/Obstacle.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Obstacle.cpp

src/CMakeFiles/RVO.dir/Obstacle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/Obstacle.cpp.i"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Obstacle.cpp > CMakeFiles/RVO.dir/Obstacle.cpp.i

src/CMakeFiles/RVO.dir/Obstacle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/Obstacle.cpp.s"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/Obstacle.cpp -o CMakeFiles/RVO.dir/Obstacle.cpp.s

src/CMakeFiles/RVO.dir/Obstacle.cpp.o.requires:

.PHONY : src/CMakeFiles/RVO.dir/Obstacle.cpp.o.requires

src/CMakeFiles/RVO.dir/Obstacle.cpp.o.provides: src/CMakeFiles/RVO.dir/Obstacle.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/RVO.dir/build.make src/CMakeFiles/RVO.dir/Obstacle.cpp.o.provides.build
.PHONY : src/CMakeFiles/RVO.dir/Obstacle.cpp.o.provides

src/CMakeFiles/RVO.dir/Obstacle.cpp.o.provides.build: src/CMakeFiles/RVO.dir/Obstacle.cpp.o


src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o: src/RVOSimulator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RVO.dir/RVOSimulator.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/RVOSimulator.cpp

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/RVOSimulator.cpp.i"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/RVOSimulator.cpp > CMakeFiles/RVO.dir/RVOSimulator.cpp.i

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/RVOSimulator.cpp.s"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/RVOSimulator.cpp -o CMakeFiles/RVO.dir/RVOSimulator.cpp.s

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.requires:

.PHONY : src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.requires

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.provides: src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/RVO.dir/build.make src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.provides.build
.PHONY : src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.provides

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.provides.build: src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o


# Object files for target RVO
RVO_OBJECTS = \
"CMakeFiles/RVO.dir/Agent.cpp.o" \
"CMakeFiles/RVO.dir/KdTree.cpp.o" \
"CMakeFiles/RVO.dir/Obstacle.cpp.o" \
"CMakeFiles/RVO.dir/RVOSimulator.cpp.o"

# External object files for target RVO
RVO_EXTERNAL_OBJECTS =

src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/Agent.cpp.o
src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/KdTree.cpp.o
src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/Obstacle.cpp.o
src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o
src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/build.make
src/libRVO.so.2.0.3: src/CMakeFiles/RVO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libRVO.so"
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RVO.dir/link.txt --verbose=$(VERBOSE)
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && $(CMAKE_COMMAND) -E cmake_symlink_library libRVO.so.2.0.3 libRVO.so.2 libRVO.so

src/libRVO.so.2: src/libRVO.so.2.0.3
	@$(CMAKE_COMMAND) -E touch_nocreate src/libRVO.so.2

src/libRVO.so: src/libRVO.so.2.0.3
	@$(CMAKE_COMMAND) -E touch_nocreate src/libRVO.so

# Rule to build all files generated by this target.
src/CMakeFiles/RVO.dir/build: src/libRVO.so

.PHONY : src/CMakeFiles/RVO.dir/build

src/CMakeFiles/RVO.dir/requires: src/CMakeFiles/RVO.dir/Agent.cpp.o.requires
src/CMakeFiles/RVO.dir/requires: src/CMakeFiles/RVO.dir/KdTree.cpp.o.requires
src/CMakeFiles/RVO.dir/requires: src/CMakeFiles/RVO.dir/Obstacle.cpp.o.requires
src/CMakeFiles/RVO.dir/requires: src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o.requires

.PHONY : src/CMakeFiles/RVO.dir/requires

src/CMakeFiles/RVO.dir/clean:
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src && $(CMAKE_COMMAND) -P CMakeFiles/RVO.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/RVO.dir/clean

src/CMakeFiles/RVO.dir/depend:
	cd /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/src/CMakeFiles/RVO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/RVO.dir/depend

