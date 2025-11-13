cmake_minimum_required(VERSION 3.15)
project(add_module)

# This should now work perfectly because python3.12-dev is installed
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development)

# And because the Python above is found, it will know where to find pybind11
find_package(pybind11 REQUIRED)

# This part remains the same
pybind11_add_module(add_module 
		add.cpp
) #<-- Make sure this is your C++ filename

# MUST RUN IN cmake -Dpybind11_DIR=`python -m pybind11 --cmakedir` ..

