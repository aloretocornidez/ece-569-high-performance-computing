# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/labs

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir

# Include any dependencies generated for this target.
include CMakeFiles/VectorAdd_Solution.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/VectorAdd_Solution.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/VectorAdd_Solution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VectorAdd_Solution.dir/flags.make

CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o: /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/labs/hw2/VectorAdd/solution.cu
CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o: CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o.depend
CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o: CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o.Release.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o"
	cd /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd && /usr/bin/cmake -E make_directory /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/.
	cd /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/./VectorAdd_Solution_generated_solution.cu.o -D generated_cubin_file:STRING=/home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/./VectorAdd_Solution_generated_solution.cu.o.cubin.txt -P /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o.Release.cmake

# Object files for target VectorAdd_Solution
VectorAdd_Solution_OBJECTS =

# External object files for target VectorAdd_Solution
VectorAdd_Solution_EXTERNAL_OBJECTS = \
"/home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o"

VectorAdd_Solution: CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o
VectorAdd_Solution: CMakeFiles/VectorAdd_Solution.dir/build.make
VectorAdd_Solution: /opt/cuda/lib64/libcudart_static.a
VectorAdd_Solution: /usr/lib/librt.a
VectorAdd_Solution: libwb.a
VectorAdd_Solution: /opt/cuda/lib64/libcudart_static.a
VectorAdd_Solution: /usr/lib/librt.a
VectorAdd_Solution: CMakeFiles/VectorAdd_Solution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VectorAdd_Solution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VectorAdd_Solution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VectorAdd_Solution.dir/build: VectorAdd_Solution
.PHONY : CMakeFiles/VectorAdd_Solution.dir/build

CMakeFiles/VectorAdd_Solution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VectorAdd_Solution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VectorAdd_Solution.dir/clean

CMakeFiles/VectorAdd_Solution.dir/depend: CMakeFiles/VectorAdd_Solution.dir/hw2/VectorAdd/VectorAdd_Solution_generated_solution.cu.o
	cd /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/labs /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/labs /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir /home/alan-manuel/Downloads/school/569-high-performance-computing/assignments/assignment-container/build_dir/CMakeFiles/VectorAdd_Solution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VectorAdd_Solution.dir/depend
