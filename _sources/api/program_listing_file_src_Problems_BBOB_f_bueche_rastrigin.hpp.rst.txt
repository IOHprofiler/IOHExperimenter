
.. _program_listing_file_src_Problems_BBOB_f_bueche_rastrigin.hpp:

Program Listing for File f_bueche_rastrigin.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_bueche_rastrigin.hpp>` (``src/Problems/BBOB/f_bueche_rastrigin.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_BUECHE_RASTRIGIN_H
   #define _F_BUECHE_RASTRIGIN_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Bueche_Rastrigin : public IOHprofiler_problem<double> {
   public:
     Bueche_Rastrigin(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(4);
       IOHprofiler_set_problem_name("Bueche_Rastrigin");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     ~Bueche_Rastrigin() {};
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (3 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(4, this->IOHprofiler_get_instance_id());
   
       for (int i = 0; i < n; i += 2) {
         xopt[i] = fabs(xopt[i]);
       }
   
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::penalty_factor = 100.0;
       Coco_Transformation_Data::lower_bound = -5.0;
       Coco_Transformation_Data::upper_bound = 5.0;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       double tmp = 0., tmp2 = 0.;
       std::vector<double> result(1);
       int n = x.size();
   
       result[0] = 0.0;
       for (int i = 0; i < n; ++i) {
         tmp += cos(2 * coco_pi * x[i]);
         tmp2 += x[i] * x[i];
       }
       result[0] = 10.0 * ((double) (long) n - tmp) + tmp2 + 0;
   
       return result[0];
     }
     
     static Bueche_Rastrigin * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Bueche_Rastrigin(instance_id, dimension);
     }
   };
   
   #endif
