
.. _program_listing_file_src_Problems_BBOB_f_schaffers10.hpp:

Program Listing for File f_schaffers10.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_schaffers10.hpp>` (``src/Problems/BBOB/f_schaffers10.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_SCHAFFERSTEN_HPP
   #define _F_SCHAFFERSTEN_HPP
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   class Schaffers10 : public IOHprofiler_problem<double> {
   public:
     Schaffers10(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(17);
       IOHprofiler_set_problem_name("Schaffers10");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
   
     ~Schaffers10() {}
   
     const double conditioning = 10;
     
     void prepare_problem() {
       std::vector<double> xopt;
       double fopt;
       std::vector<std::vector<double> > M;
       std::vector<std::vector<double> > M1;
       std::vector<double> b;
       std::vector<double> b1;
       /* compute xopt, fopt*/
       
       int n = this->IOHprofiler_get_number_of_variables();
       const long rseed = (long) (17 + 10000 * this->IOHprofiler_get_instance_id());
       bbob2009_compute_xopt(xopt, rseed, n);
       fopt = bbob2009_compute_fopt(17, this->IOHprofiler_get_instance_id());
       
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
           double exponent = 1.0 * (int) i / ((double) (long) n - 1.0);
           M[i][j] = rot2[i][j] * pow(sqrt(conditioning), exponent);
         }
       }
       bbob2009_copy_rotation_matrix(rot1,M1,b1,n);
   
       Coco_Transformation_Data::fopt = fopt;
       Coco_Transformation_Data::xopt = xopt;
       Coco_Transformation_Data::M = M;
       Coco_Transformation_Data::b = b;
       Coco_Transformation_Data::M1 = M1;
       Coco_Transformation_Data::b1 = b1;
       Coco_Transformation_Data::penalty_factor = 10.0;
     }
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
       size_t i = 0;
       std::vector<double> result(1);
   
       /* Computation core */
       result[0] = 0.0;
       for (i = 0; i < n - 1; ++i) {
         const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
         if (std::isinf(tmp) && std::isnan(sin(50.0 * pow(tmp, 0.1))))  /* sin(inf) -> nan */
           /* the second condition is necessary to pass the integration tests under Windows and Linux */
           return tmp;
         result[0] += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
       }
       result[0] = pow(result[0] / ((double) (long) n - 1.0), 2.0);
       
       return result[0];
     }
     
     static Schaffers10 * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Schaffers10(instance_id, dimension);
     }
   };
   
   #endif
