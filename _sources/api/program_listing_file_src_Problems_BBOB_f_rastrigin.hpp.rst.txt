
.. _program_listing_file_src_Problems_BBOB_f_rastrigin.hpp:

Program Listing for File f_rastrigin.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_rastrigin.hpp>` (``src/Problems/BBOB/f_rastrigin.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_RASTRIGIN_HPP
   #define _F_RASTRIGIN_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Rastrigin : public IOHprofiler_problem<double> {
   public:
     Rastrigin(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(3);
       IOHprofiler_set_problem_name("Rastrigin");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Rastrigin() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (3 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(3, this->IOHprofiler_get_instance_id());
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       size_t i = 0;
       std::vector<double> result(1);
       int n = x.size();
       double sum1 = 0.0, sum2 = 0.0;
       
       for (i = 0; i < n; ++i) {
         sum1 += cos(2.0 * coco_pi * x[i]);
         sum2 += x[i] * x[i];
       }
       /* double check std::isinf*/
       if (std::isinf(sum2)) {
         return sum2;
       }
       result[0] = 10.0 * ((double) (long) n - sum1) + sum2;
       return result[0];
     }
   
     static Rastrigin * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Rastrigin(instance_id, dimension);
     }
   };
   
   #endif
