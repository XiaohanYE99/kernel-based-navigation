# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/new-rvo2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/new-rvo2

# Include any dependencies generated for this target.
include CMakeFiles/mainDebug.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mainDebug.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mainDebug.dir/flags.make

CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o: CMakeFiles/mainDebug.dir/flags.make
CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o: Main/mainDebug.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/new-rvo2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o -c /home/new-rvo2/Main/mainDebug.cpp

CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/new-rvo2/Main/mainDebug.cpp > CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.i

CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/new-rvo2/Main/mainDebug.cpp -o CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.s

# Object files for target mainDebug
mainDebug_OBJECTS = \
"CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o"

# External object files for target mainDebug
mainDebug_EXTERNAL_OBJECTS =

mainDebug: CMakeFiles/mainDebug.dir/Main/mainDebug.cpp.o
mainDebug: CMakeFiles/mainDebug.dir/build.make
mainDebug: libRVO.a
mainDebug: /usr/lib64/libcholmod.so
mainDebug: /usr/lib64/libamd.so
mainDebug: /usr/lib64/libcolamd.so
mainDebug: /usr/lib64/libcamd.so
mainDebug: /usr/lib64/libccolamd.so
mainDebug: /usr/local/lib/libTinyVisualizer.so
mainDebug: /usr/lib64/libmpfr.so
mainDebug: /usr/lib64/libgmp.so
mainDebug: /usr/lib64/libnuma.so
mainDebug: CMakeFiles/mainDebug.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/new-rvo2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mainDebug"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mainDebug.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mainDebug.dir/build: mainDebug

.PHONY : CMakeFiles/mainDebug.dir/build

CMakeFiles/mainDebug.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mainDebug.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mainDebug.dir/clean

CMakeFiles/mainDebug.dir/depend:
	cd /home/new-rvo2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/new-rvo2 /home/new-rvo2 /home/new-rvo2 /home/new-rvo2 /home/new-rvo2/CMakeFiles/mainDebug.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mainDebug.dir/depend

