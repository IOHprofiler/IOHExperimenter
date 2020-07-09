
.. _program_listing_file_src_Problems_BBOB_f_sphere.hpp:

Program Listing for File f_sphere.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_sphere.hpp>` (``src/Problems/BBOB/f_sphere.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_SPHERE_HPP_
   #define _F_SPHERE_HPP_
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Sphere : public IOHprofiler_problem<double> {
   public:
     Sphere(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(1);
       IOHprofiler_set_problem_name("Sphere");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Sphere() {}
   
     void prepare_problem() {
       /* compute xopt, fopt*/
       std::vector<double> xopt;
       double fopt;
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (1 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(1, this->IOHprofiler_get_instance_id());
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       size_t i = 0;
       std::vector<double> result(1);
   
       result[0] = 0.0;
       for (i = 0; i < n; ++i) {
         result[0] += x[i] * x[i];
       }
       
       return result[0];
     }
   
     static Sphere * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Sphere(instance_id, dimension);
     }
   };
   
   #endif
