
.. _program_listing_file_src_Template_IOHprofiler_transformation.h:

Program Listing for File IOHprofiler_transformation.h
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_transformation.h>` (``src/Template/IOHprofiler_transformation.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef IOHPROFILER_TRANSFORMATION_H
   #define IOHPROFILER_TRANSFORMATION_H
   
   #include "IOHprofiler_common.h"
   #include "IOHprofiler_random.h"
   #include "coco_transformation.hpp"
   
   class IOHprofiler_transformation {
   public:
     
     void variables_transformation(std::vector<int> &x, const int problem_id, const int instance_id, const std::string problem_type);
   
     void variables_transformation(std::vector<double> &x, const int problem_id, const int instance_id, const std::string problem_type);
   
     void objectives_transformation(const std::vector<int> &x, std::vector<double> &y, const int problem_id, const int instance_id, const std::string problem_type);
   
     void objectives_transformation(const std::vector<double> &x, std::vector<double> &y, const int problem_id, const int instance_id, const std::string problem_type);
     
     static int xor_compute(const int x1, const int x2);
   
     void transform_vars_xor(std::vector<int> &x, const int seed);
   
     static int sigma_compute(const std::vector<int> &x, const int pos);
   
     void transform_vars_sigma(std::vector<int> &x, const int seed);
   
     void transform_obj_scale(std::vector<double> &y, const int seed);
   
     void transform_obj_shift(std::vector<double> &y, const int seed);
   
     void transform_obj_scale(double &y, const int seed);
   
     void transform_obj_shift(double &y, const int seed);
   };
   
   #endif //IOHPROFILER_TRANSFORMATION_H
