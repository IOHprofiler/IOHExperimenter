
.. _program_listing_file_src_Problems_PBO_f_leading_ones.hpp:

Program Listing for File f_leading_ones.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_leading_ones.hpp>` (``src/Problems/PBO/f_leading_ones.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_LEADING_ONES_H
   #define _F_LEADING_ONES_H
   
   
   #include "IOHprofiler_problem.h"
   
   class LeadingOnes : public IOHprofiler_problem<int> {
   public:
     LeadingOnes(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("LeadingOnes");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
     
     ~LeadingOnes() {}
   
     double internal_evaluate(const std::vector<int> &x) {
       int n = x.size();
       int result = 0;
       for (int i = 0; i != n; ++i) {
         if(x[i] == 1)
           result = i + 1;
         else
           break;
       }
       return (double)result;
     }
     
     static LeadingOnes * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new LeadingOnes(instance_id, dimension);
     }
   };
   
   #endif
