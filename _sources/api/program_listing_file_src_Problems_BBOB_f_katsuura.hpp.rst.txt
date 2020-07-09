
.. _program_listing_file_src_Problems_BBOB_f_katsuura.hpp:

Program Listing for File f_katsuura.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_katsuura.hpp>` (``src/Problems/BBOB/f_katsuura.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_KATSUURA_H
   #define _F_KATSUURA_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Katsuura : public IOHprofiler_problem<double> {
   public:
     Katsuura(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(23);
       IOHprofiler_set_problem_name("Katsuura");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Katsuura() {}
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       const double penalty_factor = 1.0;
       /* compute xopt, fopt*/
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (23 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(23, this->IOHprofiler_get_instance_id());
       
       /* compute M and b */
       M = std::vector<std::vector<double> > (n);
       for (int i = 0; i != n; i++) {
         M[i] = std::vector<double> (n);
       }
       b = std::vector<double> (n);
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
             M[i][j] += rot1[i][k] * pow(sqrt(100.0), exponent) * rot2[k][j];
           }
         }
       }
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
       Coco_Transformation_Data::penalty_factor = penalty_factor;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       size_t i, j;
       double tmp, tmp2;
       std::vector<double> result(1);
   
       /* Computation core */
       result[0] = 1.0;
       for (i = 0; i < n; ++i) {
         tmp = 0;
         for (j = 1; j < 33; ++j) {
           tmp2 = pow(2., (double) j);
           tmp += fabs(tmp2 * x[i] - floor(tmp2 * x[i] + 0.5)) / tmp2;
         }
         tmp = 1.0 + ((double) (long) i + 1) * tmp;
         /*result *= tmp;*/ /* Wassim TODO: delete once consistency check passed*/
         result[0] *= pow(tmp, 10. / pow((double) n, 1.2));
       }
       /*result = 10. / ((double) number_of_variables) / ((double) number_of_variables)
           * (-1. + pow(result, 10. / pow((double) number_of_variables, 1.2)));*/
       result[0] = 10. / ((double) n) / ((double) n) * (-1. + result[0]);
   
       return result[0];
     }
   
     static Katsuura * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Katsuura(instance_id, dimension);
     }
   };
   
   #endif
