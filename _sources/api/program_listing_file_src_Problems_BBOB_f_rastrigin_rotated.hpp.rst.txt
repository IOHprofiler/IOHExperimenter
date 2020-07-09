
.. _program_listing_file_src_Problems_BBOB_f_rastrigin_rotated.hpp:

Program Listing for File f_rastrigin_rotated.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_rastrigin_rotated.hpp>` (``src/Problems/BBOB/f_rastrigin_rotated.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_RASTRIGIN_ROTATED_HPP
   #define _F_RASTRIGIN_ROTATED_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Rastrigin_Rotated : public IOHprofiler_problem<double> {
   public:
     Rastrigin_Rotated(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(15);
       IOHprofiler_set_problem_name("Rastrigin_Rotated");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Rastrigin_Rotated() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       std::vector<std::vector<double> > M1;
       std::vector<double> b1;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (15 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(15, this->IOHprofiler_get_instance_id());
       
       /* compute M and b */
       M = std::vector<std::vector<double> > (n);
       M1 = std::vector<std::vector<double> > (n);
       for (int i = 0; i != n; i++) {
         M[i] = std::vector<double> (n);
         M1[i] = std::vector<double> (n);
       }
       b = std::vector<double> (n);
       b1 = std::vector<double> (n);
       std::vector<std::vector<double> > rot1;
       std::vector<std::vector<double> > rot2;
       bbob2009_compute_rotation(rot1, rseed + 1000000, n);
       bbob2009_compute_rotation(rot2, rseed, n);
       for (int i = 0; i < n; ++i) {
           b[i] = 0.0;
           for (int j = 0; j < n; ++j) {
               M[i][j] = 0.0;
               for (int k = 0; k < n; ++k) {
                   double exponent = 1.0 * (int) k / ((double) (long) n - 1.0);
                   M[i][j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
               }
           }
       }
       bbob2009_copy_rotation_matrix(rot1, M1, b1, n);
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
       Coco_Transformation_Data::M1 = M1;
       Coco_Transformation_Data::b1 = b1;
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
   
     static Rastrigin_Rotated * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Rastrigin_Rotated(instance_id, dimension);
     }
   };
   
   #endif
