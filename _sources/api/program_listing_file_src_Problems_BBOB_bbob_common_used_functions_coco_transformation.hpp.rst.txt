
.. _program_listing_file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation.hpp:

Program Listing for File coco_transformation.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation.hpp>` (``src/Problems/BBOB/bbob_common_used_functions/coco_transformation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _COCO_TRANSFORMATION_HPP
   #define _COCO_TRANSFORMATION_HPP
   
   #include "suite_bbob_legacy_code.hpp"
   #include "coco_transformation_vars.hpp"
   #include "coco_transformation_objs.hpp"
   
   class Coco_Transformation_Data {
   public:
     static std::vector<double> raw_x;
     static std::vector<double> xopt;
     static std::vector<double> tmp1;
     static std::vector<double> tmp2;
     static double fopt;
     static double penalty_factor;
     static double factor;
     static double lower_bound;
     static double upper_bound;
     static std::vector<std::vector<double> > M;
     static std::vector<double> b;
     static std::vector<std::vector<double> > M1;
     static std::vector<double> b1;
     static std::vector<std::vector<double> > rot1;
     static std::vector<std::vector<double> > rot2;
     static std::vector<double> datax;
     static std::vector<double> dataxx;
     static std::vector<double> minus_one;
     static double condition;
     static long rseed;
   };
   
   
   static void coco_tranformation_vars (std::vector<double> &x, const int problem_id) {
     if (problem_id == 1) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
     } else if (problem_id == 2) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_oscillate_evaluate_function(x);
     } else if (problem_id == 3) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_oscillate_evaluate_function(x);
       transform_vars_asymmetric_evaluate_function(x,0.2);
       transform_vars_conditioning_evaluate(x,10.0);
   
     } else if (problem_id == 4) {
       Coco_Transformation_Data::raw_x = x;
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_oscillate_evaluate_function(x);
       transform_vars_brs_evaluate(x);
     } else if (problem_id == 5) {
   
     } else if (problem_id == 6) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 7) {
   
     } else if (problem_id == 8) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_scale_evaluate(x,Coco_Transformation_Data::factor);
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::minus_one);
     } else if (problem_id == 9) {
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 10) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
       transform_vars_oscillate_evaluate_function(x);
     } else if (problem_id == 11) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
       transform_vars_oscillate_evaluate_function(x);
     } else if (problem_id == 12) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
       transform_vars_asymmetric_evaluate_function(x,0.5);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 13) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 14) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 15) {
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M1,Coco_Transformation_Data::b1);
       transform_vars_oscillate_evaluate_function(x);
       transform_vars_asymmetric_evaluate_function(x,0.2);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 16) {
       Coco_Transformation_Data::raw_x = x;
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M1,Coco_Transformation_Data::b1);
       transform_vars_oscillate_evaluate_function(x);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 17) {
       Coco_Transformation_Data::raw_x = x;
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M1,Coco_Transformation_Data::b1);
       transform_vars_asymmetric_evaluate_function(x,0.5);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 18) {
       Coco_Transformation_Data::raw_x = x;
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M1,Coco_Transformation_Data::b1);
       transform_vars_asymmetric_evaluate_function(x,0.5);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 19) {
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
     } else if (problem_id == 20) {
       transform_vars_x_hat_evaluate(x,Coco_Transformation_Data::rseed);
       transform_vars_scale_evaluate(x,2);
       transform_vars_z_hat_evaluate(x,Coco_Transformation_Data::xopt);
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::tmp2);
       transform_vars_conditioning_evaluate(x,10.0);
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::tmp1);
       transform_vars_scale_evaluate(x,100);
     } else if (problem_id == 21) {
   
     } else if (problem_id == 22) {
       
     } else if (problem_id == 23) {
       Coco_Transformation_Data::raw_x = x;
       transform_vars_shift_evaluate_function(x,Coco_Transformation_Data::xopt);
       transform_vars_affine_evaluate_function(x,Coco_Transformation_Data::M,Coco_Transformation_Data::b);
     } else if (problem_id == 24) {
   
     } else {
     }
   }
   
   static void coco_tranformation_objs (const std::vector<double> &x, std::vector<double> &y, const int problem_id) {
     if (problem_id == 1) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 2) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 3) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 4) {
       /* ignore large-scale test */
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
       transform_obj_penalize_evaluate(Coco_Transformation_Data::raw_x,Coco_Transformation_Data::lower_bound,Coco_Transformation_Data::upper_bound,Coco_Transformation_Data::penalty_factor,y);
         } else if (problem_id == 5) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 6) {
       transform_obj_oscillate_evaluate(y);
       transform_obj_power_evaluate(y,0.9);
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 7) {
       
     } else if (problem_id == 8) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 9) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 10) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 11) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 12) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 13) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 14) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 15) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 16) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
       transform_obj_penalize_evaluate(Coco_Transformation_Data::raw_x,Coco_Transformation_Data::lower_bound,Coco_Transformation_Data::upper_bound,Coco_Transformation_Data::penalty_factor,y);
     } else if (problem_id == 17) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
       transform_obj_penalize_evaluate(Coco_Transformation_Data::raw_x,Coco_Transformation_Data::lower_bound,Coco_Transformation_Data::upper_bound,Coco_Transformation_Data::penalty_factor,y);
     } else if (problem_id == 18) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
       transform_obj_penalize_evaluate(Coco_Transformation_Data::raw_x,Coco_Transformation_Data::lower_bound,Coco_Transformation_Data::upper_bound,Coco_Transformation_Data::penalty_factor,y);
     } else if (problem_id == 19) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 20) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 21) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 22) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else if (problem_id == 23) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
       transform_obj_penalize_evaluate(Coco_Transformation_Data::raw_x,Coco_Transformation_Data::lower_bound,Coco_Transformation_Data::upper_bound,Coco_Transformation_Data::penalty_factor,y);
     } else if (problem_id == 24) {
       transform_obj_shift_evaluate_function(y,Coco_Transformation_Data::fopt);
     } else {}
   }
   
   #endif //_COCO_TRANSFORMATION_HPP
