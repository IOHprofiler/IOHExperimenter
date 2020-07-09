
.. _program_listing_file_src_Problems_BBOB_f_griewank_rosenbrock.hpp:

Program Listing for File f_griewank_rosenbrock.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_griewank_rosenbrock.hpp>` (``src/Problems/BBOB/f_griewank_rosenbrock.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_GRIEWANK_ROSENBROCK_H
   #define _F_GRIEWANK_ROSENBROCK_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Griewank_RosenBrock : public IOHprofiler_problem<double> {
   public:
     Griewank_RosenBrock(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(19);
       IOHprofiler_set_problem_name("Griewank_RosenBrock");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Griewank_RosenBrock() {}
   
     double scales;
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (19 + 10000 * this->IOHprofiler_get_instance_id());
       fopt = bbob2009_compute_fopt(19, this->IOHprofiler_get_instance_id());
       xopt = std::vector<double>(n);
       for (int i = 0; i < n; ++i) {
         xopt[i] = -0.5;
       }
       
       /* compute M and b */
       M = std::vector<std::vector<double> > (n);
       for (int i = 0; i != n; i++) {
         M[i] = std::vector<double> (n);
       }
       b = std::vector<double> (n);
       std::vector<std::vector<double> > rot1;
       bbob2009_compute_rotation(rot1, rseed, n);
       scales = 1. > (sqrt((double) n) / 8.) ? 1. : (sqrt((double) n) / 8.);
       for (int i = 0; i < n; ++i) {
         for (int j = 0; j < n; ++j) {
           rot1[i][j] *= scales;
         }
       }
       bbob2009_copy_rotation_matrix(rot1,M,b,n);
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
   
       // double tmp;
       // std::vector<double> tmp_best_variables(n,0);
       // for (int j = 0; j < n; ++j) {
       //   tmp = 0;
       //   for (int i = 0; i < n; ++i) {
       //     tmp += rot1[i][j];
       //   }
       //   tmp_best_variables[j] = tmp / (2. * scales);
       // }
       // IOHprofiler_set_best_variables(tmp_best_variables);
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       size_t i = 0;
       double tmp = 0;
       std::vector<double> result(1);
       int n = x.size();
   
       /* Computation core */
       result[0] = 0.0;
       for (i = 0; i < n - 1; ++i) {
         const double c1 = x[i] * x[i] - x[i + 1];
         const double c2 = 1.0 - x[i];
         tmp = 100.0 * c1 * c1 + c2 * c2;
         result[0] += tmp / 4000. - cos(tmp);
       }
       result[0] = 10. + 10. * result[0] / (double) (n - 1);
   
       return result[0];
     }
   
     static Griewank_RosenBrock * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Griewank_RosenBrock(instance_id, dimension);
     }
   };
   
   #endif
