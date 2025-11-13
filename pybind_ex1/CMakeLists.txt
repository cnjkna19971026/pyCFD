cmake_minimum_required(VERSION 3.15)
project(cpp_solver_module)

# This should now work perfectly because python3.12-dev is installed
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development)

# And because the Python above is found, it will know where to find pybind11
find_package(pybind11 REQUIRED)

# This part remains the same
pybind11_add_module(cpp_solver_module src/pybind_reference.cpp) #<-- Make sure this is your C++ filename

# MUST RUN IN cmake -Dpybind11_DIR=`python -m pybind11 --cmakedir` ..


#cmake_minimum_required(VERSION 3.15)
#project(grid_solver)
#
#find_package(python3.12 REQUIRED COMPONENTS Interpreter Development.Module)
#
#
## This command uses the ACTIVE Python environment to find pybind11.
## It automatically figures out the include path for you.
#find_package(pybind11 REQUIRED)
#
## This uses the information from find_package to set up the compiler.
## Let's say your file is named grid_test.cpp
#pybind11_add_module(grid_solver grid_test.cpp)
