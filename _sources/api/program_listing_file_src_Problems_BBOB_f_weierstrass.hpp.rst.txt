
.. _program_listing_file_src_Problems_BBOB_f_weierstrass.hpp:

Program Listing for File f_weierstrass.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_weierstrass.hpp>` (``src/Problems/BBOB/f_weierstrass.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_WEIERSTRASS_HPP
   #define _F_WEIERSTRASS_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   #define F_WEIERSTRASS_SUMMANDS 12
   
   class Weierstrass : public IOHprofiler_problem<double> {
   public:
     Weierstrass(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(16);
       IOHprofiler_set_problem_name("Weierstrass");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     
     ~Weierstrass() {}
   
     const double condition = 100.0;
     double f0;
     double ak[F_WEIERSTRASS_SUMMANDS];
     double bk[F_WEIERSTRASS_SUMMANDS];
     
     void prepare_problem() {
   
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       std::vector<std::vector<double> > M1;
       std::vector<double> b1;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (16 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(16, this->IOHprofiler_get_instance_id());
       
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
             const double base = 1.0 / sqrt(condition);
             const double exponent = 1.0 * (int) k / ((double) (long) n - 1.0);
             M[i][j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
           }
         }
       }
       bbob2009_copy_rotation_matrix(rot1, M1, b1, n);
   
       f0 = 0.0;
       for (int i = 0; i < F_WEIERSTRASS_SUMMANDS; ++i) {
         ak[i] = pow(0.5, (double) i);
         bk[i] = pow(3., (double) i);
         f0 += ak[i] * cos(2 * coco_pi * bk[i] * 0.5);
       }
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
       Coco_Transformation_Data::M1 = M1;
       Coco_Transformation_Data::b1 = b1;
       Coco_Transformation_Data::penalty_factor = 10.0 / (double) n;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       size_t i, j;
   
       std::vector<double> result(1);
   
       result[0] = 0.0;
       for (i = 0; i < n; ++i) {
         for (j = 0; j < F_WEIERSTRASS_SUMMANDS; ++j) {
           result[0] += cos(2 * coco_pi * (x[i] + 0.5) * bk[j]) * ak[j];
         }
       }
       result[0] = 10.0 * pow(result[0] / (double) (long) n - f0, 3.0);
   
       return result[0];
     }
   
     static Weierstrass * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Weierstrass(instance_id, dimension);
     }
   };
   
   #endif
