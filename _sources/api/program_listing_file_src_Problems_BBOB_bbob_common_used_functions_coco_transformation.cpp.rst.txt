
.. _program_listing_file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation.cpp:

Program Listing for File coco_transformation.cpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_bbob_common_used_functions_coco_transformation.cpp>` (``src/Problems/BBOB/bbob_common_used_functions/coco_transformation.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #include "coco_transformation.hpp"
   
   std::vector<double> Coco_Transformation_Data::raw_x = std::vector<double>(0);
   std::vector<double> Coco_Transformation_Data::xopt = std::vector<double>(0);
   std::vector<double> Coco_Transformation_Data::tmp1 = std::vector<double>(0);
   std::vector<double> Coco_Transformation_Data::tmp2 = std::vector<double>(0);
   double Coco_Transformation_Data::fopt = 0;
   double Coco_Transformation_Data::penalty_factor = 0;
   double Coco_Transformation_Data::factor = 0;
   double Coco_Transformation_Data::lower_bound = 0;
   double Coco_Transformation_Data::upper_bound = 0;
   std::vector<std::vector<double> > Coco_Transformation_Data::M = std::vector<std::vector<double>>(0);
   std::vector<double> Coco_Transformation_Data::b = std::vector<double>(0);;
   std::vector<std::vector<double> > Coco_Transformation_Data::M1 = std::vector<std::vector<double>>(0);
   std::vector<double> Coco_Transformation_Data::b1 = std::vector<double>(0);;
   std::vector<std::vector<double> > Coco_Transformation_Data::rot1 = std::vector<std::vector<double>>(0);
   std::vector<std::vector<double> > Coco_Transformation_Data::rot2 = std::vector<std::vector<double>>(0);
   std::vector<double> Coco_Transformation_Data::datax = std::vector<double>(0);;
   std::vector<double> Coco_Transformation_Data::dataxx = std::vector<double>(0);;
   std::vector<double> Coco_Transformation_Data::minus_one = std::vector<double>(0);;
   double Coco_Transformation_Data::condition = 0;
   long Coco_Transformation_Data::rseed = 0;
   
   
   
   
