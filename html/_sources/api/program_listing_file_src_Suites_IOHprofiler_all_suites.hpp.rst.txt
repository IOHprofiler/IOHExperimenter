
.. _program_listing_file_src_Suites_IOHprofiler_all_suites.hpp:

Program Listing for File IOHprofiler_all_suites.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Suites_IOHprofiler_all_suites.hpp>` (``src/Suites/IOHprofiler_all_suites.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_ALL_SUITES_HPP
   #define _IOHPROFILER_ALL_SUITES_HPP
   
   #include "IOHprofiler_PBO_suite.hpp"
   #include "IOHprofiler_BBOB_suite.hpp"
   
   #include "IOHprofiler_class_generator.h"
   
   static registerInFactory<IOHprofiler_suite<int>,PBO_suite> regPBO("PBO");
   static registerInFactory<IOHprofiler_suite<double>,BBOB_suite> regBBOB("BBOB");
   
   #endif //_IOHPROFILER_ALL_SUITES_HPP
