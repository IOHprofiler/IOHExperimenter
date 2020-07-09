
.. _program_listing_file_src_Problems_BBOB_f_rosenbrock.hpp:

Program Listing for File f_rosenbrock.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_rosenbrock.hpp>` (``src/Problems/BBOB/f_rosenbrock.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ROSENBROCK_HPP
   #define _F_ROSENBROCK_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Rosenbrock : public IOHprofiler_problem<double> {
   public:
     Rosenbrock(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(8);
       IOHprofiler_set_problem_name("Rosenbrock");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Rosenbrock() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       std::vector<double> minus_one;
       double fopt, factor;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (8 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(8, this->IOHprofiler_get_instance_id());
   
       for (int i = 0; i < n; ++i) {
         minus_one.push_back(-1.0);
         xopt[i] *= 0.75;
       }
   
       factor = 1.0 > (sqrt((double) n) / 8.0) ? 1 : (sqrt((double) n) / 8.0);
       Coco_Transformation_Data::minus_one = minus_one;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::factor = factor;
       Coco_Transformation_Data::fopt = fopt;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       size_t i = 0;
       std::vector<double> result(1);
       double s1 = 0.0, s2 = 0.0, tmp;
   
       for (i = 0; i < n - 1; ++i) {
         tmp = (x[i] * x[i] - x[i + 1]);
         s1 += tmp * tmp;
         tmp = (x[i] - 1.0);
         s2 += tmp * tmp;
       }
       result[0] = 100.0 * s1 + s2;
   
       return result[0];
     }
   
     static Rosenbrock * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Rosenbrock(instance_id, dimension);
     }
   };
   
   #endif
