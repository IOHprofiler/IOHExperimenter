
.. _program_listing_file_src_Template_IOHprofiler_common.h:

Program Listing for File IOHprofiler_common.h
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_common.h>` (``src/Template/IOHprofiler_common.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_COMMON_HPP
   #define _IOHPROFILER_COMMON_HPP
   
   #include "IOHprofiler_platform.h"
   
   #include <iostream>
   #include <fstream>
   #include <sstream>
   #include <string>
   #include <vector>
   #include <cassert>
   #include <cmath>
   #include <cstdlib> 
   #include <cstddef>
   #include <limits>
   #include <cfloat>
   #include <algorithm>
   #include <memory>
   #include <map>
   
   #define IOHprofiler_PATH_MAX 200
   
   #define IOHprofiler_MAX_DIMENSION 20000
   
   #define DEFAULT_PROBLEM_ID 0
   
   #define DEFAULT_INSTANCE 1
   
   #define DEFAULT_DIMENSION 4
   
   #define MAX_BUFFER_SIZE 65534
   
   enum IOH_optimization_type {Minimization=0, Maximization=1};
   
   void IOH_error(std::string error_info);
   
   void IOH_warning(std::string warning_info);
   
   void IOH_log(std::string log_info);
   
   void IOH_log(std::string log_info, std::ofstream &log_stream);
   
   
   template<class valueType>
   void copyVector(const std::vector<valueType> v1, std::vector<valueType> &v2) {
     v2.assign(v1.begin(),v1.end());
   }
   
   template<class valueType>
   bool compareVector(const std::vector<valueType> &v1, const std::vector<valueType> &v2) {
     int n = v1.size();
     if(n != v2.size()){
       IOH_error("Two compared vector must be with the same size\n");
       return false;
     }
     for (int i = 0; i != n; ++i)
     {
       if(v1[i] != v2[i]){
         return false;
       }
     }
     return true;
   }
   
   template<class valueType>
   bool compareObjectives(const std::vector<valueType> &v1, const std::vector<valueType> &v2, const IOH_optimization_type optimization_type){
     int n = v1.size();
     if(n != v2.size()){
       IOH_error("Two compared objective vector must be with the same size\n");
       return false;
     }
     if (optimization_type == IOH_optimization_type::Maximization) {
       for (int i = 0; i != n; ++i)
       {
         if (v1[i] <= v2[i]) {
           return false;
         }
       }
       return true;
     } else {
       for (int i = 0; i != n; ++i)
       {
         if (v1[i] >= v2[i]) {
           return false;
         }
       }
       return true;
     }
   }
   
   template<class valueType>
   bool compareObjectives(const valueType v1, const valueType v2, const IOH_optimization_type optimization_type){
     if (optimization_type == IOH_optimization_type::Maximization) {
       if (v1 <= v2){
         return false;
       }
       return true;
     } else {
       if (v1 >= v2) {
         return false;
       }
       return true;
     }
   }
   
   template<class valueType> std::string _toString(const valueType v) {
     std::ostringstream ss;
     ss << v;
     return ss.str();
   }
   
   #endif //_IOHPROFILER_COMMON_HPP
