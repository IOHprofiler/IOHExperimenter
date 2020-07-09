
.. _program_listing_file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation_vars.hpp:

Program Listing for File coco_transformation_vars.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation_vars.hpp>` (``src/Problems/BBOB/bbob_common_used_functions/coco_transformation_vars.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _COCO_TRANSFORMATION_VARS_HPP
   #define _COCO_TRANSFORMATION_VARS_HPP
   
   #include "suite_bbob_legacy_code.hpp"
   
   static void transform_vars_affine_evaluate_function(std::vector<double> &x, 
                                                       const std::vector<std::vector<double> > &M,
                                                       const std::vector<double> &b) {
     size_t i, j;
     int number_of_variables = x.size();
     double *cons_values;
     int is_feasible;
     std::vector<double> temp_x = x;
     for (i = 0; i < number_of_variables; ++i) {
       /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
       x[i] = b[i];
       for (j = 0; j < number_of_variables; ++j) {
         x[i] += temp_x[j] * M[i][j];
       }
     }
   }
   
   static void transform_vars_asymmetric_evaluate_function(std::vector <double> &x, const double beta) {
     size_t i;
     double exponent;
     int number_of_variables = x.size();
     for (i = 0; i < number_of_variables; ++i) {
       if (x[i] > 0.0) {
         exponent = 1.0
             + ((beta * (double) (long) i) / ((double) (long) number_of_variables - 1.0)) * sqrt(x[i]);
         x[i] = pow(x[i], exponent);
       } else {
         x[i] = x[i];
       }
     }
   }
   
   static void transform_vars_brs_evaluate(std::vector<double> &x) {
     size_t i;
     double factor;
     int number_of_variables = x.size();
   
     for (i = 0; i < number_of_variables; ++i) {
       /* Function documentation says we should compute 10^(0.5 *
        * (i-1)/(D-1)). Instead we compute the equivalent
        * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
        */
       factor = pow(sqrt(10.0), (double) (long) i / ((double) (long) number_of_variables - 1.0));
       /* Documentation specifies odd indices and starts indexing
        * from 1, we use all even indices since C starts indexing
        * with 0.
        */
       if (x[i] > 0.0 && i % 2 == 0) {
         factor *= 10.0;
       }
       x[i] = factor * x[i];
     }
   }
   
   static void transform_vars_conditioning_evaluate(std::vector<double> &x, const double alpha) {
     size_t i;
     int number_of_variables = x.size();
   
     for (i = 0; i < number_of_variables; ++i) {
       /* OME: We could precalculate the scaling coefficients if we
        * really wanted to.
        */
       x[i] = pow(alpha, 0.5 * (double) (long) i / ((double) (long) number_of_variables - 1.0))
           * x[i];
     }
   }
   
   static void transform_vars_oscillate_evaluate_function(std::vector<double> &x) {
     static const double alpha = 0.1;
     int number_of_variables = x.size();
     double tmp, base;
     size_t i;
     
     for (i = 0; i < number_of_variables; ++i) {
       if (x[i] > 0.0) {
         tmp = log(x[i]) / alpha;
         base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
         x[i] = pow(base, alpha);
       } else if (x[i] < 0.0) {
         tmp = log(-x[i]) / alpha;
         base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
         x[i] = -pow(base, alpha);
       } else {
         x[i] = 0.0;
       }
     }
   }
   
   static void transform_vars_scale_evaluate(std::vector<double> &x, const double factor) {
     size_t i;
     int number_of_variables = x.size();
   
     for (i = 0; i < number_of_variables; ++i) {
       x[i] = factor * x[i];
     }
   }
   
   static void transform_vars_shift_evaluate_function(std::vector<double> &x, const std::vector<double> &offset) {
     size_t i;
     int number_of_variables = x.size();
    
     for (i = 0; i < number_of_variables; ++i) {
       x[i] = x[i] - offset[i];
     }
   }
   
   static void transform_vars_x_hat_evaluate(std::vector<double> &x, const long seed) {
     size_t i;
     int number_of_variables = x.size();
     std::vector<double> tmp_x;
   
   
     bbob2009_unif(tmp_x, number_of_variables, seed);
   
     for (i = 0; i < number_of_variables; ++i) {
       if (tmp_x[i] < 0.5) {
         x[i] = -x[i];
       } else {
         x[i] = x[i];
       }
     }
   }
   
   static void transform_vars_x_hat_generic_evaluate(std::vector<double> &x, const std::vector<double> &sign_vector) {
     size_t i;
     int number_of_variables = x.size();
   
     for (i = 0; i < number_of_variables; ++i) {
       x[i] = 2.0 * sign_vector[i] * x[i];
     }
   }
   
   static void transform_vars_z_hat_evaluate(std::vector<double> &x,  const std::vector<double> &xopt) {
     size_t i;
     int number_of_variables = x.size();
     std::vector<double> temp_x = x;
     x[0] = temp_x[0];
   
     for (i = 1; i < number_of_variables; ++i) {
       x[i] = temp_x[i] + 0.25 * (temp_x[i - 1] - 2.0 * fabs(xopt[i - 1]));
     }
   }
   
   #endif
