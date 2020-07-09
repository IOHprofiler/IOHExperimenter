
.. _program_listing_file_src_Problems_BBOB_f_lunacek_bi_rastrigin.hpp:

Program Listing for File f_lunacek_bi_rastrigin.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_lunacek_bi_rastrigin.hpp>` (``src/Problems/BBOB/f_lunacek_bi_rastrigin.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_LUNACEK_BI_RASTRIGIN_HPP
   #define _F_LUNACEK_BI_RASTRIGIN_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Lunacek_Bi_Rastrigin : public IOHprofiler_problem<double> {
   public:
     Lunacek_Bi_Rastrigin(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(24);
       IOHprofiler_set_problem_name("Lunacek_Bi_Rastrigin");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Lunacek_Bi_Rastrigin() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       std::vector<std::vector<double> > rot1;
       std::vector<std::vector<double> > rot2;
       double fopt;
       std::vector<double> tmpvect;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (24 + 10000 * this->IOHprofiler_get_instance_id());
       
       fopt = bbob2009_compute_fopt(24, this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       bbob2009_compute_rotation(rot1, rseed + 1000000, n);
       bbob2009_compute_rotation(rot2, rseed, n);
   
       Coco_Transformation_Data::fopt = fopt;
   
       Coco_Transformation_Data::rot1 = rot1;
       Coco_Transformation_Data::rot2 = rot2;
   
     
       bbob2009_gauss(tmpvect, n, rseed);
       for (int i = 0; i < n; ++i) {
         xopt[i] = 0.5 * 2.5;
         if (tmpvect[i] < 0.0) {
           xopt[i] *= -1.0;
         }
       }
       Coco_Transformation_Data::xopt = xopt;
       IOHprofiler_set_best_variables(xopt);
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       std::vector<double> result(1);
       int n = x.size();
       static const double condition = 100.;
       size_t i, j;
       double penalty = 0.0;
       static const double mu0 = 2.5;
       static const double d = 1.;
       const double s = 1. - 0.5 / (sqrt((double) (n + 20)) - 4.1);
       const double mu1 = -sqrt((mu0 * mu0 - d) / s);
       double sum1 = 0., sum2 = 0., sum3 = 0.;
       std::vector<double> tmpvect(n);
       std::vector<double> x_hat(n);
       std::vector<double> z(n);
   
       for (i = 0; i < n; ++i) {
         double tmp;
         tmp = fabs(x[i]) - 5.0;
         if (tmp > 0.0)
           penalty += tmp * tmp;
       }
   
       /* x_hat */
       for (i = 0; i < n; ++i) {
         x_hat[i] = 2. * x[i];
         if (Coco_Transformation_Data::xopt[i] < 0.) {
           x_hat[i] *= -1.;
         }
       }
       /* affine transformation */
       for (i = 0; i < n; ++i) {
         double c1;
         tmpvect[i] = 0.0;
         c1 = pow(sqrt(condition), ((double) i) / (double) (n - 1));
         for (j = 0; j < n; ++j) {
           tmpvect[i] += c1 * Coco_Transformation_Data::rot2[i][j] * (x_hat[j] - mu0);
         }
       }
       for (i = 0; i < n; ++i) {
         z[i] = 0;
         for (j = 0; j < n; ++j) {
           z[i] += Coco_Transformation_Data::rot1[i][j] * tmpvect[j];
         }
       }
       /* Computation core */
       for (i = 0; i < n; ++i) {
         sum1 += (x_hat[i] - mu0) * (x_hat[i] - mu0);
         sum2 += (x_hat[i] - mu1) * (x_hat[i] - mu1);
         sum3 += cos(2 * coco_pi * z[i]);
       }
       result[0] = std::min(sum1, d * (double) n + s * sum2)
           + 10. * ((double) n - sum3) + 1e4 * penalty;
       return result[0];
     }
   
     static Lunacek_Bi_Rastrigin * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Lunacek_Bi_Rastrigin(instance_id, dimension);
     }
   };
   
   #endif
