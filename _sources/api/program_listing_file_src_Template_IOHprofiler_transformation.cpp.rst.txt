
.. _program_listing_file_src_Template_IOHprofiler_transformation.cpp:

Program Listing for File IOHprofiler_transformation.cpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_transformation.cpp>` (``src/Template/IOHprofiler_transformation.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   
   #include "IOHprofiler_transformation.h"
   
   void IOHprofiler_transformation::variables_transformation(std::vector<int> &x, const int problem_id, const int instance_id, const std::string problem_type) { 
     if (problem_type == "pseudo_Boolean_problem") {
       if (instance_id > 1 && instance_id <= 50) { 
         this->transform_vars_xor(x,instance_id);
       } else if (instance_id > 50 && instance_id <= 100) {
         this->transform_vars_sigma(x,instance_id);
       }
     }
   }
   
   void IOHprofiler_transformation::variables_transformation(std::vector<double> &x, const int problem_id, const int instance_id, const std::string problem_type) {
     if (problem_type == "bbob") {
       coco_tranformation_vars(x,problem_id);
     }
   }
   
   void IOHprofiler_transformation::objectives_transformation(const std::vector<int> &x, std::vector<double> &y, const int problem_id, const int instance_id, const std::string problem_type) {
     if (problem_type == "pseudo_Boolean_problem") {
       if (instance_id > 1) {
         this->transform_obj_scale(y,instance_id);
         this->transform_obj_shift(y,instance_id);
       }
     }
   }
   
   void IOHprofiler_transformation::objectives_transformation(const std::vector<double> &x, std::vector<double> &y, const int problem_id, const int instance_id, const std::string problem_type) {
     if (problem_type == "bbob") {
       coco_tranformation_objs(x,y,problem_id);
     }
   }
   
   int IOHprofiler_transformation::xor_compute(const int x1, const int x2){
     return (int) (x1 != x2);
   }
   
   void IOHprofiler_transformation::transform_vars_xor(std::vector<int> &x, const int seed) {
     std::vector<double> random_x;
     int xor_value;
     int n = x.size();
     IOHprofiler_random::IOHprofiler_uniform_rand(n,seed, random_x);
     for (int i = 0; i < n; ++i) {
       xor_value = (int)(2 * floor(1e4 * random_x[i]) / 1e4);
       x[i] = xor_compute(x[i],xor_value);
     }
   }
   
   int IOHprofiler_transformation::sigma_compute(const std::vector<int> &x, const int pos) {
     return x[pos];
   }
   
   void IOHprofiler_transformation::transform_vars_sigma(std::vector<int> &x, const int seed) {
     std::vector<int> copy_x;
     std::vector<int> index;
     std::vector<double> random_x;
     int N = x.size(),t,temp;
     
     copy_x = x;
   
     index.reserve(N);
     for (int i = 0; i != N; ++i) {
       index.push_back(i);
     }
   
     IOHprofiler_random::IOHprofiler_uniform_rand(N,seed,random_x);
     for (int i = 0; i != N; ++i) {
       t = (int)floor(random_x[i] * N);
       temp = index[0];
       index[0] = index[t];
       index[t] = temp;
     }
     for (int i = 0; i < N; ++i) {
       x[i] = sigma_compute(copy_x,index[i]);
     }
   }
   
   void IOHprofiler_transformation::transform_obj_scale(std::vector<double> &y, const int seed) {
     std::vector<double> scale;
     IOHprofiler_random::IOHprofiler_uniform_rand(1,seed,scale);
     scale[0] = scale[0] * 1e4 / 1e4 * 4.8 + 0.2;
     for (int i = 0; i < y.size(); ++i) {
       y[i] = y[i] * scale[0];
     }
   }
   
   void IOHprofiler_transformation::transform_obj_shift(std::vector<double> &y, const int seed) {
     std::vector<double> shift;
     IOHprofiler_random::IOHprofiler_uniform_rand(1,seed,shift);
     shift[0] = shift[0] * 1e4 / 1e4 * 2000 - 1000;
     for (int i = 0; i < y.size(); ++i) {
       y[i] = y[i] + shift[0];
     }
   }
   
   void IOHprofiler_transformation::transform_obj_scale(double &y, const int seed) {
     std::vector<double> scale;
     IOHprofiler_random::IOHprofiler_uniform_rand(1,seed,scale);
     scale[0] = scale[0] * 1e4 / 1e4 * 4.8 + 0.2;
     y = y * scale[0];
   }
   
   void IOHprofiler_transformation::transform_obj_shift(double &y, const int seed) {
     std::vector<double> shift;
     IOHprofiler_random::IOHprofiler_uniform_rand(1,seed,shift);
     shift[0] = shift[0] * 1e4 / 1e4 * 2000 - 1000;
     y = y + shift[0];
   }
