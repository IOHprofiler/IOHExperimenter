
.. _program_listing_file_src_Problems_PBO_f_labs.hpp:

Program Listing for File f_labs.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_labs.hpp>` (``src/Problems/PBO/f_labs.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_LABS_H
   #define _F_LABS_H
   
   #include "IOHprofiler_problem.h"
   
   class LABS : public IOHprofiler_problem<int> {
   public:
     LABS(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("LABS");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
     
     ~LABS() {}
   
     double correlation(const std::vector<int> x, const int n, int k)
     {
       int x1,x2;
       double result;
       result = 0.0;
       for (int i = 0 ; i < n - k; ++i) {
           if (x[i] == 0) {
             x1 = -1;
           } else {
             x1 = 1;
           }
           if (x[i + k] == 0) {
             x2 = -1;
           } else {
             x2 = 1;
           }
           result += x1 * x2;
       }
       return result;
     }
   
     double internal_evaluate(const std::vector<int> &x) {
   
       int n = x.size();
       double result = 0.0, cor;
       for (int k = 1; k != n; ++k) {
         cor = correlation(x,n,k);
         result += cor * cor;
       }
       result = (double)(n*n)/2.0/result;
       return (double)result;
     }
     
     static LABS * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new LABS(instance_id, dimension);
     }
   };
   
   #endif
