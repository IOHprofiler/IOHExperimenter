
.. _program_listing_file_src_Problems_PBO_f_linear.hpp:

Program Listing for File f_linear.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_linear.hpp>` (``src/Problems/PBO/f_linear.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_LINEAR_H
   #define _F_LINEAR_H
   
   #include "IOHprofiler_problem.h"
   
   class Linear : public IOHprofiler_problem<int> {
   public:
     Linear(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
   
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("Linear");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       Initilize_problem(dimension);
     }
     
     ~Linear() {}
   
     void Initilize_problem(int dimension) {
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       int n = x.size();
       double result = 0;
       for (int i = 0; i < n; ++i) {
         result += (double)x[i] * (double)(i + 1);
       }
       return (double)result;
     }
   
     static Linear * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Linear(instance_id, dimension);
     }
   };
   
   #endif
