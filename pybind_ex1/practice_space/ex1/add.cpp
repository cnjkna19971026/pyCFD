
#include <pybind11/pybind11.h>

namespace py = pybind11;

double add(double i , double j)
{
	return i+j;
}


PYBIND11_MODULE(add_module, m) {

	// The 'add_module' name MUST match the target name in your CMakeLists.txt
    // 'm' is the variable that represents the module object

    m.doc() = "A C++ particle simulation module for Python";

    // MODIFIED: Expose the new function signature with named arguments and defaults
    m.def(
		"add",           // The name of the function in Python
        &add,            // The address of your C++ function
        "A function that adds two numbers" // Optional docstring for the function
	);
}
