
.. _program_listing_file_src_Problems_BBOB_f_ellipsoid.hpp:

Program Listing for File f_ellipsoid.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_ellipsoid.hpp>` (``src/Problems/BBOB/f_ellipsoid.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ELLIPSOID_H
   #define _F_ELLIPSOID_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Ellipsoid : public IOHprofiler_problem<double> {
   public:
     Ellipsoid(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(2);
       IOHprofiler_set_problem_name("Ellipsoid");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     
     ~Ellipsoid() {}
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (2 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(2, this->IOHprofiler_get_instance_id());
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
     }
   
   
     double internal_evaluate(const std::vector<double> &x) {
       static const double condition = 1.0e6;
       size_t i = 0;
       int n = x.size();
       std::vector<double> result(1);
   
   
       result[0] = x[i] * x[i];
       for (i = 1; i < n; ++i) {
         const double exponent = 1.0 * (double) (long) i / ((double) (long) n - 1.0);
         result[0] += pow(condition, exponent) * x[i] * x[i];
       }
   
       return result[0];
     }
   
     static Ellipsoid * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Ellipsoid(instance_id, dimension);
     }
   };
   
   #endif
