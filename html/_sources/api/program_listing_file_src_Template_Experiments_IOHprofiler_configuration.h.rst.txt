
.. _program_listing_file_src_Template_Experiments_IOHprofiler_configuration.h:

Program Listing for File IOHprofiler_configuration.h
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_Experiments_IOHprofiler_configuration.h>` (``src/Template/Experiments/IOHprofiler_configuration.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_CONFIGURATION_H
   #define _IOHPROFILER_CONFIGURATION_H
   
   #include "IOHprofiler_string.hpp"
   
   #define MAXLINESIZE 1024
   #define MAXKEYNUMBER 100
   
   static int max_number_of_problem;
   static int max_dimension;
   
   class _Dict {
   public:
     _Dict() {};
     ~_Dict() {};
     int n = 0;
     size_t size;
     std::vector<std::string> section;
     std::vector<std::string> value;
     std::vector<std::string> key;
   };
   
   typedef enum _LINE_ {
     EMPTY,
     COMMENT,
     SECTION,
     VALUE,
     CON_ERROR
   } linecontent;
   
   class IOHprofiler_configuration {
   public:
     IOHprofiler_configuration() {}
   
     int set_Dict(_Dict &dict, const std::string section, const std::string key, const std::string value);
   
     std::string get_Dict_String(const _Dict dict, const std::string section, const std::string key);
   
     std::vector<int> get_Dict_int_vector(const _Dict dict, const std::string section, const std::string key, const int _min, const int _max);
   
     int get_Dict_Int(const _Dict dict, const std::string section, const std::string key);
   
     bool get_Dict_bool(const _Dict dict, const std::string section, const std::string key);
   
     linecontent add_Line(const std::string input_line, std::string &section, std::string &key, std::string &value);
   
     _Dict load(const std::string filename);
   
     void readcfg(std::string filename);
   
     std::string get_suite_name();
   
     std::vector<int> get_problem_id();
   
     std::vector<int> get_instance_id();
   
     std::vector<int> get_dimension();
   
     std::string get_output_directory();
   
     std::string get_result_folder();
   
     std::string get_algorithm_info();
   
     std::string get_algorithm_name();
   
     bool get_complete_triggers();
   
     bool get_update_triggers();
   
     std::vector<int> get_base_evaluation_triggers();
   
     int get_number_target_triggers();
   
     int get_number_interval_triggers();
   
   private:
     std::string cfgFile = "configuration.ini";
     std::string suite_name;
     std::vector<int> problem_id;
     std::vector<int> instance_id;
     std::vector<int> dimension;
     std::string output_directory;
     std::string result_folder;
     std::string algorithm_info;
     std::string algorithm_name;
     bool complete_triggers;
     bool update_triggers;
     std::vector<int> base_evaluation_triggers;
     int number_target_triggers;
     int number_interval_triggers;
   };
   
   #endif
