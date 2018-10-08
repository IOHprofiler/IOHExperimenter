/**
 * @file suite_IOHprofiler.c
 * @brief Implementation of the IOHprofiler suite containing 24 noiseless single-objective functions in 6
 * dimensions.
 */

#include "IOHprofiler.h"

#include "f_binary.c"
#include "f_jump.c"
#include "f_leading_ones.c"
#include "f_linear.c"
#include "f_one_max.c"
#include "f_labs.c"

static IOHprofiler_suite_t *IOHprofiler_suite_allocate(const char *suite_name,
                                                       const size_t number_of_functions,
                                                       const size_t number_of_dimensions,
                                                       const size_t *dimensions,
                                                       const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the IOHprofiler suite.
 */
static IOHprofiler_suite_t *suite_PBO_initialize(void) {
    IOHprofiler_suite_t *suite;
    size_t i;
    const char *suite_name = "PBO";
    const size_t number_of_functions = 5000;
    /*const size_t dimensions[] = {47, 500, 1000, 1500, 2000, 2500, 3000}; 
    */
    size_t dimensions[5000];

    const size_t number_of_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);
    for (i = 0; i < number_of_dimensions; ++i){
        dimensions[i] = i + 1;
    }
    /* IMPORTANT: Make sure to change the default instance for every new workshop! */
    suite = IOHprofiler_suite_allocate(suite_name, number_of_functions, number_of_dimensions, dimensions, "year: 2018");

    return suite;
}

/**
 * @brief Sets the instances associated with years for the IOHprofiler suite.
 */
static const char *suite_PBO_get_instances_by_year(const int year) {
    if (year == 2018) {
        return "1-100";
    } else {
        IOHprofiler_error("suite_PBO_get_instances_by_year(): year %d not defined for suite_IOHprofiler", year);
        return NULL;
    }
}

/**
 * @brief Creates and returns a IOHprofiler problem without needing the actual IOHprofiler suite.
 *
 * Useful for other suites as well (see for example suite_biobj.c).
 */
static IOHprofiler_problem_t *PBO_get_problem(const size_t function,
                                              const size_t dimension,
                                              const size_t instance) {
    IOHprofiler_problem_t *problem = NULL;

    const char *problem_id_template = "PBO_f%03lu_i%02lu_d%02lu";
    const char *problem_name_template = "PBO suite problem f%lu instance %lu in %luD";

    const long rseed = (long)(function + 10000 * instance);

    if (function == 1) {
        problem = f_one_max_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                         problem_id_template, problem_name_template);
    } else if (function == 2) {
        problem = f_leading_ones_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                              problem_id_template, problem_name_template);
    } else if (function == 3) {
        problem = f_jump_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                      problem_id_template, problem_name_template);
    } else if (function == 4) {
        problem = f_linear_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                        problem_id_template, problem_name_template);
    } else if (function == 5) {
        problem = f_labs_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                        problem_id_template, problem_name_template);
    } else if (function == 5) {
        problem = f_binary_IOHprofiler_problem_allocate(function, dimension, instance, rseed,
                                                        problem_id_template, problem_name_template);
    } else {
        IOHprofiler_error("get_IOHprofiler_problem(): cannot retrieve problem f%lu instance %lu in %luD",
                          (unsigned long)function, (unsigned long)instance, (unsigned long)dimension);
        return NULL; /* Never reached */
    }

    return problem;
}

/**
 * @brief Returns the problem from the IOHprofiler suite that corresponds to the given parameters.
 *
 * @param suite The IOHprofiler suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static IOHprofiler_problem_t *suite_PBO_get_problem(IOHprofiler_suite_t *suite,
                                                    const size_t function_idx,
                                                    const size_t dimension_idx,
                                                    const size_t instance_idx) {
    IOHprofiler_problem_t *problem = NULL;

    const size_t function = suite->functions[function_idx];
    const size_t dimension = suite->dimensions[dimension_idx];
    const size_t instance = suite->instances[instance_idx];

    problem = PBO_get_problem(function, dimension, instance);

    problem->suite_dep_function = function;
    problem->suite_dep_instance = instance;
    problem->suite_dep_index = IOHprofiler_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

    return problem;
}
