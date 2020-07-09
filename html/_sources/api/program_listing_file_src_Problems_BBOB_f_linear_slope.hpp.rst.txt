
.. _program_listing_file_src_Problems_BBOB_f_linear_slope.hpp:

Program Listing for File f_linear_slope.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_linear_slope.hpp>` (``src/Problems/BBOB/f_linear_slope.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_LINEAR_SLOPE_H
   #define _F_LINEAR_SLOPE_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Linear_Slope : public IOHprofiler_problem<double> {
   public:
     Linear_Slope(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(5);
       IOHprofiler_set_problem_name("Linear_Slope");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Linear_Slope() {}
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (5 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(5, this->IOHprofiler_get_instance_id());
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
   
       std::vector<double> tmp_best_variables = std::vector<double>(n);
       for (int i = 0; i < n; ++i) {
         if (xopt[i] < 0.0) {
           tmp_best_variables[i] = this->IOHprofiler_get_lowerbound()[i];
         } else {
           tmp_best_variables[i] = this->IOHprofiler_get_upperbound()[i];
         }
       }
       this->IOHprofiler_set_best_variables(tmp_best_variables);
     }
     
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       static const double alpha = 100.0;
       size_t i;
       std::vector<double> result(1);
       
       for (i = 0; i < n; ++i) {
         double base, exponent, si;
   
         base = sqrt(alpha);
         exponent = (double) (long) i / ((double) (long) n - 1);
         if (Coco_Transformation_Data::xopt[i] > 0.0) {
           si = pow(base, exponent);
         } else {
           si = -pow(base, exponent);
         }
         /* boundary handling */
         if (x[i] * Coco_Transformation_Data::xopt[i] < 25.0) {
           result[0] += 5.0 * fabs(si) - si * x[i];
         } else {
           result[0] += 5.0 * fabs(si) - si * Coco_Transformation_Data::xopt[i];
         }
       }
   
       return result[0];
     }
   
     static Linear_Slope * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Linear_Slope(instance_id, dimension);
     }
   };
   
   #endif
