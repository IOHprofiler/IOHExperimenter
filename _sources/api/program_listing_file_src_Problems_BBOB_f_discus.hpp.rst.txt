
.. _program_listing_file_src_Problems_BBOB_f_discus.hpp:

Program Listing for File f_discus.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_discus.hpp>` (``src/Problems/BBOB/f_discus.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_DISCUS_H
   #define _F_DISCUS_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Discus : public IOHprofiler_problem<double> {
   public:
     Discus(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(11);
       IOHprofiler_set_problem_name("Discus");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0.0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     
     ~Discus() {}
   
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<double> b;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (11 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(11, this->IOHprofiler_get_instance_id());
       
       /* compute M and b */
       M = std::vector<std::vector<double> > (n);
       for (int i = 0; i != n; i++) {
         M[i] = std::vector<double> (n);
       }
       b = std::vector<double> (n);
       std::vector<std::vector<double> > rot1;
       bbob2009_compute_rotation(rot1, rseed + 1000000, n);
       bbob2009_copy_rotation_matrix(rot1,M,b,n);
       
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       static const double condition = 1.0e6;
       std::vector<double> result(1);
       int n = x.size();
   
   
       result[0] = condition * x[0] * x[0];
       for (int i = 1; i < n; ++i) {
         result[0] += x[i] * x[i];
       }
   
       return result[0];
     }
     
     static Discus * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Discus(instance_id, dimension);
     }
   };
   
   #endif
