#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "mylib.h"





namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;


// typedef Eigen::MatrixXd Matrix;

// typedef Matrix::Scalar Scalar;
// constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;


PYBIND11_MODULE(MyLib, m) {
    m.doc() = "optional module docstring";

    py::class_<NearSolver>(m, "NearSolver")
    .def(py::init<int,int,int,std::string>())  
    
    //.def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
    .def("get_a", &NearSolver::getA, py::return_value_policy::reference_internal)
    .def("get_b", &NearSolver::getB, py::return_value_policy::reference_internal)
    .def("generate_matrices", &NearSolver::generateMatrices)
    .def("set_camera_matrix",&NearSolver::setCameraMatrix)
    .def("get_x", &NearSolver::getX, py::return_value_policy::reference_internal)
    .def("get_uc", &NearSolver::getUc, py::return_value_policy::reference_internal)
    //.def("get_valid_lights", &NearSolver::getValidLights)
    //.def("get_visible_pixels", &NearSolver::getVisiblePixels)
    .def("attenuateMatrix", &NearSolver::attenuateM, py::return_value_policy::reference_internal)
    

    

    


    // .def("get_sparse", &MyClass::getSparse,py::return_value_policy::reference_internal)
    // .def("set_matrix", &MyClass::setMatrix)
    


    
    ;



}