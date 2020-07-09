
.. _program_listing_file_src_Template_Experiments_IOHprofiler_experimenter.h:

Program Listing for File IOHprofiler_experimenter.h
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_Experiments_IOHprofiler_experimenter.h>` (``src/Template/Experiments/IOHprofiler_experimenter.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_EXPERIMENTER_HPP
   #define _IOHPROFILER_EXPERIMENTER_HPP
   
   #include "IOHprofiler_common.h"
   #include "IOHprofiler_all_suites.hpp"
   #include "IOHprofiler_csv_logger.h"
   #include "IOHprofiler_configuration.h"
   
   #include <ctime>
   
   template <class InputType> class IOHprofiler_experimenter {
   public:
     typedef void _algorithm(std::shared_ptr<IOHprofiler_problem<InputType> >, std::shared_ptr<IOHprofiler_csv_logger> logger);
   
     IOHprofiler_experimenter();
     IOHprofiler_experimenter(std::string configFileName, _algorithm *algorithm);
     IOHprofiler_experimenter(IOHprofiler_suite<InputType> suite, std::shared_ptr<IOHprofiler_csv_logger> csv_logger, _algorithm * algorithm);
   
     ~IOHprofiler_experimenter(){};
   
     void _run();
     
     void _set_independent_runs(int n);
   
     void print_info(std::string info);
   
     std::string vectorToString(std::vector<int> v);
   
   private:
     IOHprofiler_configuration conf;
     std::shared_ptr<IOHprofiler_suite<InputType> > configSuite;
     std::shared_ptr<IOHprofiler_problem<InputType> > current_problem;
     std::shared_ptr<IOHprofiler_csv_logger> config_csv_logger;
     int independent_runs = 1;
   
     _algorithm *algorithm;
   };
   
   #include "IOHprofiler_experimenter.hpp"
   
   #endif //_IOHPROFILER_EXPERIMENTER_H
