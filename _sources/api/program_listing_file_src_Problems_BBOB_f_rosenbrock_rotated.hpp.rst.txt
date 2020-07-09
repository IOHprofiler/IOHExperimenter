
.. _program_listing_file_src_Problems_BBOB_f_rosenbrock_rotated.hpp:

Program Listing for File f_rosenbrock_rotated.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_rosenbrock_rotated.hpp>` (``src/Problems/BBOB/f_rosenbrock_rotated.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ROSENBROCK_ROTATED_HPP
   #define _F_ROSENBROCK_ROTATED_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Rosenbrock_Rotated : public IOHprofiler_problem<double> {
   public:
     Rosenbrock_Rotated(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(9);
       IOHprofiler_set_problem_name("Rosenbrock_Rotated");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Rosenbrock_Rotated() {}
   
     void prepare_problem() {
       double fopt, factor;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (9 + 10000 * this->IOHprofiler_get_instance_id());
       fopt = bbob2009_compute_fopt(9, this->IOHprofiler_get_instance_id());
       
       factor = 1.0 > (sqrt((double) n) / 8.0) ? 1 : (sqrt((double) n) / 8.0);
       /* compute M and b */
       M = std::vector<std::vector<double> > (n);
       for (int i = 0; i != n; i++) {
         M[i] = std::vector<double> (n);
       }
       b = std::vector<double> (n);
       std::vector<std::vector<double> > rot1;
       bbob2009_compute_rotation(rot1, rseed, n);
       for (int row = 0; row < n; ++row) {
         for (int column = 0; column < n; ++column) {
           M[row][column] = factor * rot1[row][column];
         }
         b[row] = 0.5;
       }
   
       // std::vector<double> tmp_best_variables(n,0.0);
       // for (int column = 0; column < n; ++column) { /* Wassim: manually set xopt = rot1^T ones(dimension)/(2*factor) */
       //   double tmp = 0;
       //   for (int row = 0; row < n; ++row) {
       //     tmp += rot1[row][column];
       //   }
       //   tmp_best_variables[column] = tmp / (2. * factor);
       // }
       // IOHprofiler_set_best_variables(tmp_best_variables);
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::factor = factor;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
       
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
   
     static Rosenbrock_Rotated * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Rosenbrock_Rotated(instance_id, dimension);
     }
   };
   
   #endif
