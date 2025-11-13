#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>


using namespace std;

namespace py = pybind11;

// a simple struct to represent a 2d vector. in a real cfd code, this would
// be a more complex, high-performance class.
struct vector2d {
    double x = 0.0;
    double y = 0.0;
};

// define type aliases for clarity, a common practice in large codebases.
using scalarField = vector<vector<double>>;
using vectorField = vector<vector<vector2d>>;

/**
  @brief computes the gradient of a scalar field using central differences.
  the gradient of a scalar t is a vector: ∇t = (∂t/∂x, ∂t/∂y).
  @param field the input scalar field (e.g., pressure, temperature).
  @param dx the grid spacing in the x-direction.
  @param dy the grid spacing in the y-direction.
  @return a vector field representing the gradient.
 */
vectorField compute_gradient(const scalarField& field, double dx, double dy) {
    if (field.empty() || field[0].empty()) {
        return {};
    }
    size_t rows = field.size();
    size_t cols = field[0].size();
    vectorField gradient(rows, vector<vector2d>(cols));

    // iterate over interior points. boundaries are a special case we'll ignore here for simplicity.
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // central difference for ∂t/∂x: (t(i,j+1) - t(i,j-1)) / 2dx
            gradient[i][j].x = (field[i][j + 1] - field[i][j - 1]) / (2.0 * dx);
            
            // central difference for ∂t/∂y: (t(i+1,j) - t(i-1,j)) / 2dy
            gradient[i][j].y = (field[i + 1][j] - field[i - 1][j]) / (2.0 * dy);
        }
    }
    return gradient;
}

void printVectorField(const vectorField& field )
{
  	//if (field.empty() || field[0].empty()) {
	//
	//}

	cout<<"----------"<<"vectorField :"<<"----------"<<endl;
	size_t rows = field.size();
	size_t cols = field[0].size();

	for (size_t i = 0 ; i< rows; ++i)
	{
		cout<<"| ";
		for(size_t j =0 ; j< cols; ++j)
		{
			if(j<cols-1)
				cout<<setw(5)<<"( "<<field[i][j].x <<"," << field[i][j].y <<" )";
			else
			{
				cout<<setw(5)<<"( "<<field[i][j].x <<"," << field[i][j].y <<" )";
				cout<<setw(5)<<" |"<<endl;
			}
		}
		
	}
}

void printScalarField(const scalarField& field )
{
	cout<<"----------"<<"scalarField :"<<"----------"<<endl;

	size_t rows = field.size();
	size_t cols = field[0].size();

	for (size_t i = 0 ; i< rows; ++i)
	{
		cout<<"| ";
		for(size_t j =0 ; j< cols; ++j)
		{
			if(j<cols-1)
				cout<<setw(4)<<field[i][j] ;
			else
			{
				cout<<setw(4)<<field[i][j] ;
				cout<<setw(4)<<" |"<<endl;
			}
		}
		
	}
}

/**
  @brief Computes the divergence of a vector field using central differences.
  The divergence of a vector V=(u,v) is a scalar: ∇·V = ∂u/∂x + ∂v/∂y.
  @param field The input vector field (e.g., velocity).
  @param dx The grid spacing in the x-direction.
  @param dy The grid spacing in the y-direction.
  @return A scalar field representing the divergence.
 */
scalarField compute_divergence(const vectorField& field, double dx, double dy) {
    if (field.empty() || field[0].empty()) {
        return {};
    }
    size_t rows = field.size();
    size_t cols = field[0].size();
    scalarField divergence(rows, vector<double>(cols, 0.0));
    
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // Central difference for ∂u/∂x
            double du_dx = (field[i][j + 1].x - field[i][j - 1].x) / (2.0 * dx);

            // Central difference for ∂v/∂y
            double dv_dy = (field[i + 1][j].y - field[i - 1][j].y) / (2.0 * dy);
            
            divergence[i][j] = du_dx + dv_dy;
        }
    }
    return divergence;
}

void write_vtk(const string& filename, const scalarField& scalar_data, const vectorField& vector_data, double dx, double dy) 
{
    if (scalar_data.empty() || scalar_data[0].empty()) {
        cout << "Error: Cannot write empty data to VTK file." << endl;
        return;
    }

    size_t rows = scalar_data.size();
    size_t cols = scalar_data[0].size();
    size_t num_points = rows * cols;

    ofstream vtk_file(filename);
    if (!vtk_file.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // --- VTK Header ---
    vtk_file << "# vtk DataFile Version 3.0" << endl;
    vtk_file << "CFD Gradient Calculation Results" << endl;
    vtk_file << "ASCII" << endl;
    vtk_file << "DATASET STRUCTURED_POINTS" << endl;
    
    // --- Grid Geometry ---
    // VTK is 3D, so we represent our 2D grid as a 1-layer-deep 3D grid.
    vtk_file << "DIMENSIONS " << cols << " " << rows << " " << 1 << endl;
    vtk_file << "ORIGIN 0 0 0" << endl;
    vtk_file << "SPACING " << dx << " " << dy << " " << 1.0 << endl;

    // --- Data Associated with Points ---
    vtk_file << "POINT_DATA " << num_points << endl;

    // --- Write Scalar Data (Temperature) ---
    vtk_file << "SCALARS Temperature double 1" << endl;
    vtk_file << "LOOKUP_TABLE default" << endl;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            vtk_file << scalar_data[i][j] << " ";
        }
        vtk_file << endl;
    }

    // --- Write Vector Data (Gradient) ---
    // VTK vectors are always 3-component, so we add a zero for the z-component.
    vtk_file << "VECTORS Gradient double" << endl;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            vtk_file << vector_data[i][j].x << " " << vector_data[i][j].y << " 0.0" << endl;
        }
    }

    vtk_file.close();
    cout << "Successfully wrote results to " << filename << endl;
}
 


PYBIND11_MODULE(cpp_solver_module, m)
{
	m.doc() = "pybind11 implement";
	
	py::class_<vector2d>(m,"vector2d")
		.def(py::init<>())
		.def_readwrite("x", &vector2d::x)	
		.def_readwrite("y", &vector2d::y);

	m.def("run sim.", &compute_gradient,"calculating compute_gradient", 
			py::arg(),			

					
					)
}


int main() {

	double dx = 1.0;
	double dy = 1.0;

    // Example: A 5x5 grid with 1.0 spacing.
    scalarField temperature(10, vector<double>(10));

	printScalarField(temperature);
	cout<<"\n"<<"----------------------------"<<"\n";

    // Create a "hot spot" in the middle
    temperature[2][2] = 100.0;




    vectorField grad = compute_gradient(temperature, 1.0, 1.0);

	printScalarField(temperature);
	cout<<"\n"<<"----------------------------"<<"\n";

	printVectorField(grad);
	cout<<"\n"<<"----------------------------"<<"\n";

   //  The gradient vector at (2,1) should point right (towards the hot spot).
   //  The gradient vector at (1,2) should point up (towards the hot spot).
   cout << "Gradient at (2,1): (" << grad[2][1].x << ", " << grad[2][1].y << ")" << endl;
   cout << "Gradient at (1,2): (" << grad[1][2].x << ", " << grad[1][2].y << ")" << endl;

//	write_vtk("test.vtk",temperature,grad,dx,dy);

}
