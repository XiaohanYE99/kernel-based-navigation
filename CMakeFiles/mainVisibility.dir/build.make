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
CMAKE_SOURCE_DIR = /home/yxhan/yxh/kernel-based-navigation-new-rvo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yxhan/yxh/kernel-based-navigation-new-rvo

# Include any dependencies generated for this target.
include CMakeFiles/mainVisibility.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mainVisibility.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mainVisibility.dir/flags.make

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o: CMakeFiles/mainVisibility.dir/flags.make
CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o: Main/mainVisibility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yxhan/yxh/kernel-based-navigation-new-rvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o -c /home/yxhan/yxh/kernel-based-navigation-new-rvo/Main/mainVisibility.cpp

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yxhan/yxh/kernel-based-navigation-new-rvo/Main/mainVisibility.cpp > CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.i

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yxhan/yxh/kernel-based-navigation-new-rvo/Main/mainVisibility.cpp -o CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.s

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.requires:

.PHONY : CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.requires

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.provides: CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.requires
	$(MAKE) -f CMakeFiles/mainVisibility.dir/build.make CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.provides.build
.PHONY : CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.provides

CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.provides.build: CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o


# Object files for target mainVisibility
mainVisibility_OBJECTS = \
"CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o"

# External object files for target mainVisibility
mainVisibility_EXTERNAL_OBJECTS =

mainVisibility: CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o
mainVisibility: CMakeFiles/mainVisibility.dir/build.make
mainVisibility: libRVO.a
mainVisibility: /usr/lib/x86_64-linux-gnu/libcholmod.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libamd.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libcolamd.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libcamd.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libccolamd.so
mainVisibility: /usr/local/lib/libTinyVisualizer.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libmpfr.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libgmp.so
mainVisibility: /usr/lib/x86_64-linux-gnu/libnuma.so
mainVisibility: CMakeFiles/mainVisibility.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yxhan/yxh/kernel-based-navigation-new-rvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mainVisibility"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mainVisibility.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mainVisibility.dir/build: mainVisibility

.PHONY : CMakeFiles/mainVisibility.dir/build

CMakeFiles/mainVisibility.dir/requires: CMakeFiles/mainVisibility.dir/Main/mainVisibility.cpp.o.requires

.PHONY : CMakeFiles/mainVisibility.dir/requires

CMakeFiles/mainVisibility.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mainVisibility.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mainVisibility.dir/clean

CMakeFiles/mainVisibility.dir/depend:
	cd /home/yxhan/yxh/kernel-based-navigation-new-rvo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yxhan/yxh/kernel-based-navigation-new-rvo /home/yxhan/yxh/kernel-based-navigation-new-rvo /home/yxhan/yxh/kernel-based-navigation-new-rvo /home/yxhan/yxh/kernel-based-navigation-new-rvo /home/yxhan/yxh/kernel-based-navigation-new-rvo/CMakeFiles/mainVisibility.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mainVisibility.dir/depend

