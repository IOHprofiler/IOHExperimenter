
.. _program_listing_file_src_Problems_BBOB_f_sharp_ridge.hpp:

Program Listing for File f_sharp_ridge.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_sharp_ridge.hpp>` (``src/Problems/BBOB/f_sharp_ridge.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_SHARP_RIDGE_HPP
   #define _F_SHARP_RIDGE_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Sharp_Ridge : public IOHprofiler_problem<double> {
   public:
     Sharp_Ridge(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(13);
       IOHprofiler_set_problem_name("Sharp_Ridge");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Sharp_Ridge() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (13 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(13, this->IOHprofiler_get_instance_id());
       
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
             M[i][j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
           }
         }
       }
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       static const double alpha = 100.0;
       const double vars_40 = 1; /* generalized: number_of_variables <= 40 ? 1 : number_of_variables / 40.0; */
       size_t i = 0;
       std::vector<double> result(1);
   
       
       result[0] = 0.0;
       for (i = (size_t)(ceil(vars_40)); i < n; ++i) {
         result[0] += x[i] * x[i];
       }
       result[0] = alpha * sqrt(result[0] / vars_40);
       for (i = 0; i < (size_t)ceil(vars_40); ++i) {
         result[0] += x[i] * x[i] / vars_40;
       }
   
       return result[0];
     }
   
     static Sharp_Ridge * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Sharp_Ridge(instance_id, dimension);
     }
   };
   
   #endif
