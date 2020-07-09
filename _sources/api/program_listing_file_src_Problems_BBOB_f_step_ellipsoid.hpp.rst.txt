
.. _program_listing_file_src_Problems_BBOB_f_step_ellipsoid.hpp:

Program Listing for File f_step_ellipsoid.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_step_ellipsoid.hpp>` (``src/Problems/BBOB/f_step_ellipsoid.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_STEP_ELLIPSOID_HPP
   #define _F_STEP_ELLIPSOID_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Step_Ellipsoid : public IOHprofiler_problem<double> {
   public:
     Step_Ellipsoid(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(7);
       IOHprofiler_set_problem_name("Step_Ellipsoid");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     
     ~Step_Ellipsoid() {}
   
     std::vector<double> xopt;
     double fopt;
     std::vector<std::vector<double> > rot1;
     std::vector<std::vector<double> > rot2;
     std::vector<double> datax,dataxx;
     
     void prepare_problem() {
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (7 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(7, this->IOHprofiler_get_instance_id());
       
   
       bbob2009_compute_rotation(rot1, rseed + 1000000, n);
       bbob2009_compute_rotation(rot2, rseed, n);
   
       datax = std::vector<double>(n);
       dataxx = std::vector<double>(n);
   
       IOHprofiler_set_best_variables(xopt);
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       static const double condition = 100;
       static const double alpha = 10.0;
       size_t i, j;
       double penalty = 0.0, x1;
       std::vector<double> result(1);
   
       for (i = 0; i < n; ++i) {
         double tmp;
         tmp = fabs(x[i]) - 5.0;
         if (tmp > 0.0)
           penalty += tmp * tmp;
       }
       
       for (i = 0; i < n; ++i) {
         double c1;
         datax[i] = 0.0;
         c1 = sqrt(pow(condition / 10., (double) i / (double) (n - 1)));
         for (j = 0; j < n; ++j) {
           datax[i] += c1 * rot2[i][j] * (x[j] - xopt[j]);
         }
       }
       x1 = datax[0];
       
       for (i = 0; i < n; ++i) {
         if (fabs(datax[i]) > 0.5) /* TODO: Documentation: no fabs() in documentation */
           datax[i] = (double)floor(datax[i] + 0.5);
         else
           datax[i] = (double)floor(alpha * datax[i] + 0.5) / alpha;
       }
       
       for (i = 0; i < n; ++i) {
         dataxx[i] = 0.0;
         for (j = 0; j < n; ++j) {
           dataxx[i] += rot1[i][j] * datax[j];
         }
       }
       
       /* Computation core */
       result[0] = 0.0;
       for (i = 0; i < n; ++i) {
         double exponent;
         exponent = (double) (long) i / ((double) (long) n - 1.0);
         result[0] += pow(condition, exponent) * dataxx[i] * dataxx[i];
         ;
       }
       result[0] = 0.1 * ((fabs(x1) * 1.0e-4) > result[0] ? (fabs(x1) * 1.0e-4) : result[0]) + penalty + fopt;
       
       return result[0];
     }
   
     static Step_Ellipsoid * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Step_Ellipsoid(instance_id, dimension);
     }
   };
   
   #endif
