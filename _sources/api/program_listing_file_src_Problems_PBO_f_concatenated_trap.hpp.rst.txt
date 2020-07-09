
.. _program_listing_file_src_Problems_PBO_f_concatenated_trap.hpp:

Program Listing for File f_concatenated_trap.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_concatenated_trap.hpp>` (``src/Problems/PBO/f_concatenated_trap.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_CONCATENATED_TRAP_H
   #define _F_CONCATENATED_TRAP_H
   
   #include "IOHprofiler_problem.h"
   
   class Concatenated_Trap : public IOHprofiler_problem<int> {
   public:
   
     Concatenated_Trap(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("Concatenated_Trap");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_optimal((double)dimension);
     }
     
     ~Concatenated_Trap() {};
   
     int k = 5;
   
     double internal_evaluate(const std::vector<int> &x) {
       int n = x.size();
       double result = 0.0;
       double block_result;
   
       int m = n / k;
       for (int i = 1; i <= m; ++i) {
         block_result = 0.0;
         for (int j = i*k - k; j != i*k; ++j) {
           block_result += x[j];
         }
         if (block_result == k) {
           result += 1;
         } else {
           result += ((double)(k - 1 - block_result) / (double) k );
         }
       }
       int remain_k = n - m * k;
       if (remain_k != 0 ) {
         block_result = 0.0;
         for (int j = m * k - 1; j != n; ++j) {
           block_result += x[j];
         }
         if (block_result == remain_k) {
           result += 1;
         } else {
           result += ((double)(remain_k - 1 - block_result) / (double) remain_k );
         }
       }
   
       return result;
     }
   
     static Concatenated_Trap * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Concatenated_Trap(instance_id, dimension);
     }
   };
   
   #endif
