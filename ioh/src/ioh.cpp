#include <pybind11/pybind11.h>
#include "ioh.hpp"
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

struct Pet {
    Pet(const std::string &name) : name(name) { }
    std::string name;
};

struct Dog : Pet {
    Dog(const std::string &name) : Pet(name) { }
    std::string bark() const { return "woof!"; }
};

template <class InputType>
class Custom_IOH_Function : public ioh::problem::base<InputType> {
public:
    /* Inherit the constructors */
    using ioh::problem::base<InputType>::base;

    double internal_evaluate(const std::vector<InputType>& x) {
        PYBIND11_OVERRIDE_PURE(double, ioh::problem::base<InputType>, internal_evaluate, x);
    }
    /* Trampoline (need one for each virtual function) */
    // double internal_evaluate(const std::vector<double>& x) override {
    //     PYBIND11_OVERRIDE_PURE(double, ioh::problem::base<double>, internal_evaluate, std::vector<double>);
    // }
};


// class Custom_IOH_Function : public ioh::problem::bbob::bbob_base {
// public:
//     /* Inherit the constructors */
//     using ioh::problem::bbob::bbob_base::bbob_base;

//     double internal_evaluate(const std::vector<double>& x) {
//         PYBIND11_OVERRIDE(double, ioh::problem::bbob::bbob_base, internal_evaluate, x);
//     }
//     /* Trampoline (need one for each virtual function) */
//     // double internal_evaluate(const std::vector<double>& x) override {
//     //     PYBIND11_OVERRIDE_PURE(double, ioh::problem::base<double>, internal_evaluate, std::vector<double>);
//     // }
// };

PYBIND11_MODULE(iohcpp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("to_lower", &ioh::common::to_lower, "make a string to lowercase");

    // py::class_<ioh::problem::base<double>>(m, "double_function");

    // BBOB Functions //

    py::class_<ioh::problem::bbob::bbob_base>(m, "bbob_function")
        .def("evaluate", &ioh::problem::bbob::bbob_base::evaluate)
        .def("reset", &ioh::problem::bbob::bbob_base::reset_problem)
        .def("best_so_far_raw_objectives", &ioh::problem::bbob::bbob_base::get_best_so_far_raw_objectives)
        .def("evaluations", &ioh::problem::bbob::bbob_base::get_evaluations)
        .def("number_of_variables", &ioh::problem::bbob::bbob_base::get_number_of_variables)
        .def("type", &ioh::problem::bbob::bbob_base::get_problem_type)
        .def("logger_info", &ioh::problem::bbob::bbob_base::loggerCOCOInfo);

        py::class_<ioh::problem::bbob::Sphere, ioh::problem::bbob::bbob_base>(m, "Sphere")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Ellipsoid, ioh::problem::bbob::bbob_base>(m, "Ellipsoid")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Rastrigin, ioh::problem::bbob::bbob_base>(m, "Rastrigin")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Bueche_Rastrigin, ioh::problem::bbob::bbob_base>(m, "Bueche_Rastrigin")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Linear_Slope, ioh::problem::bbob::bbob_base>(m, "Linear_Slope")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::AttractiveSector, ioh::problem::bbob::bbob_base>(m, "AttractiveSector")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Step_Ellipsoid, ioh::problem::bbob::bbob_base>(m, "Step_Ellipsoid")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Rosenbrock, ioh::problem::bbob::bbob_base>(m, "Rosenbrock")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Rosenbrock_Rotated, ioh::problem::bbob::bbob_base>(m, "Rosenbrock_Rotated")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Ellipsoid_Rotated, ioh::problem::bbob::bbob_base>(m, "Ellipsoid_Rotated")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Discus, ioh::problem::bbob::bbob_base>(m, "Discus")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Bent_Cigar, ioh::problem::bbob::bbob_base>(m, "Bent_Cigar")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Sharp_Ridge, ioh::problem::bbob::bbob_base>(m, "Sharp_Ridge")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Different_Powers, ioh::problem::bbob::bbob_base>(m, "Different_Powers")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Rastrigin_Rotated, ioh::problem::bbob::bbob_base>(m, "Rastrigin_Rotated")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Weierstrass, ioh::problem::bbob::bbob_base>(m, "Weierstrass")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Schaffers10, ioh::problem::bbob::bbob_base>(m, "Schaffers10")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Schaffers1000, ioh::problem::bbob::bbob_base>(m, "Schaffers1000")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Griewank_RosenBrock, ioh::problem::bbob::bbob_base>(m, "Griewank_RosenBrock")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Schwefel, ioh::problem::bbob::bbob_base>(m, "Schwefel")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Gallagher101, ioh::problem::bbob::bbob_base>(m, "Gallagher101")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Gallagher21, ioh::problem::bbob::bbob_base>(m, "Gallagher21")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Katsuura, ioh::problem::bbob::bbob_base>(m, "Katsuura")
            .def(py::init<int, int>());
        py::class_<ioh::problem::bbob::Lunacek_Bi_Rastrigin, ioh::problem::bbob::bbob_base>(m, "Lunacek_Bi_Rastrigin")
            .def(py::init<int, int>());


        // PBO Functions
    py::class_<ioh::problem::pbo::pbo_base>(m, "pbo_function")
        .def("evaluate", &ioh::problem::pbo::pbo_base::evaluate)
        .def("reset", &ioh::problem::pbo::pbo_base::reset_problem)
        .def("get_best_so_far_raw_objectives", &ioh::problem::pbo::pbo_base::get_best_so_far_raw_objectives)
        .def("evaluations", &ioh::problem::pbo::pbo_base::get_evaluations)
        // .def("target", &ioh::problem::pbo::pbo_base::get_target)
        .def("number_of_variables", &ioh::problem::pbo::pbo_base::get_number_of_variables)
        .def("type", &ioh::problem::pbo::pbo_base::get_problem_type)
        .def("logger_info", &ioh::problem::pbo::pbo_base::loggerInfo);

        py::class_<ioh::problem::pbo::OneMax, ioh::problem::pbo::pbo_base>(m, "OneMax")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes, ioh::problem::pbo::pbo_base>(m, "LeadingOnes")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::Linear, ioh::problem::pbo::pbo_base>(m, "Linear")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Dummy1, ioh::problem::pbo::pbo_base>(m, "OneMax_Dummy1")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Dummy2, ioh::problem::pbo::pbo_base>(m, "OneMax_Dummy2")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Neutrality, ioh::problem::pbo::pbo_base>(m, "OneMax_Neutrality")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Epistasis, ioh::problem::pbo::pbo_base>(m, "OneMax_Epistasis")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Ruggedness1, ioh::problem::pbo::pbo_base>(m, "OneMax_Ruggedness1")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Ruggedness2, ioh::problem::pbo::pbo_base>(m, "OneMax_Ruggedness2")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::OneMax_Ruggedness3, ioh::problem::pbo::pbo_base>(m, "OneMax_Ruggedness3")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Dummy1, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Dummy1")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Dummy2, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Dummy2")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Neutrality, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Neutrality")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Epistasis, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Epistasis")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Ruggedness1, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Ruggedness1")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Ruggedness2, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Ruggedness2")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LeadingOnes_Ruggedness3, ioh::problem::pbo::pbo_base>(m, "LeadingOnes_Ruggedness3")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::LABS, ioh::problem::pbo::pbo_base>(m, "LABS")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::MIS, ioh::problem::pbo::pbo_base>(m, "MIS")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::Ising_Ring, ioh::problem::pbo::pbo_base>(m, "Ising_Ring")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::Ising_Torus, ioh::problem::pbo::pbo_base>(m, "Ising_Torus")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::Ising_Triangular, ioh::problem::pbo::pbo_base>(m, "Ising_Triangular")
            .def(py::init<int, int>());
        py::class_<ioh::problem::pbo::NQueens, ioh::problem::pbo::pbo_base>(m, "NQueens")
            .def(py::init<int, int>());

        // Enable automatic conversion of string to fs::path (needed for constructor of logger)
        py::class_<fs::path>(m, "Path")
            .def(py::init<std::string>());
        py::implicitly_convertible<std::string, fs::path>();

        // Define seperate loggers for PBO and BBOB functions (will later become template-less)
        py::class_<ioh::logger::Default<ioh::problem::pbo::pbo_base>>(m, "PBO_Logger")
            .def(py::init<fs::path>())
            .def(py::init<>())
            .def("track_problem", &ioh::logger::Default<ioh::problem::pbo::pbo_base>::track_problem)
            .def("track_suite", &ioh::logger::Default<ioh::problem::pbo::pbo_base>::track_suite)
            .def("do_log", &ioh::logger::Default<ioh::problem::pbo::pbo_base>::do_log);

        py::class_<ioh::logger::Default<ioh::problem::bbob::bbob_base>>(m, "BBOB_Logger")
            .def(py::init<fs::path>())
            .def(py::init<>())
            .def("track_problem", &ioh::logger::Default<ioh::problem::bbob::bbob_base>::track_problem)
            .def("track_suite", &ioh::logger::Default<ioh::problem::bbob::bbob_base>::track_suite)
            .def("do_log", &ioh::logger::Default<ioh::problem::bbob::bbob_base>::do_log);

        py::class_<ioh::problem::base<double>, Custom_IOH_Function<double>>(m, "Custom_IOH_Function")//, py::dynamic_attr())
            .def(py::init<>())
            .def("reset", &ioh::problem::base<double>::reset_problem)
            .def("prepare", &ioh::problem::base<double>::prepare_problem)
            .def("evaluate", &ioh::problem::base<double>::evaluate)
            .def_property("fid", &ioh::problem::base<double>::get_problem_id, &ioh::problem::base<double>::set_problem_id)
            .def_property("iid", &ioh::problem::base<double>::get_instance_id, &ioh::problem::base<double>::set_instance_id)
            .def_property("name", &ioh::problem::base<double>::get_problem_name, &ioh::problem::base<double>::set_problem_name)
            .def_property("type", &ioh::problem::base<double>::get_problem_type, &ioh::problem::base<double>::set_problem_type)
            // .def_property("lowerbound", &ioh::problem::base<double>::get_lowerbound, &ioh::problem::base<double>::set_lowerbound)
            // .def_property("upperbound", &ioh::problem::base<double>::get_upperbound, &ioh::problem::base<double>::set_upperbound)
            // .def_property("dim", &ioh::problem::base<double>::get_number_of_variables, py::overload_cast<int>(&ioh::problem::base<double>::set_number_of_variables))
            .def_property("dim", &ioh::problem::base<double>::get_number_of_variables, &ioh::problem::base<double>::set_number_of_variables)
            .def("internal_evaluate", &ioh::problem::base<double>::internal_evaluate);

        // py::class_<ioh::problem::bbob::bbob_base, Custom_IOH_Function>(m, "Custom_IOH_Function")
        //     .def(py::init<int, std::string>())
        //     .def("internal_evaluate", &ioh::problem::bbob::bbob_base::internal_evaluate);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

