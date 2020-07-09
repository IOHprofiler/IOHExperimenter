
.. _program_listing_file_src_Template_IOHprofiler_suite.h:

Program Listing for File IOHprofiler_suite.h
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_suite.h>` (``src/Template/IOHprofiler_suite.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_SUITE_H
   #define _IOHPROFILER_SUITE_H
   
   #include "IOHprofiler_class_generator.h"
   #include "IOHprofiler_problem.h"
   
   typedef std::map<std::string, int> PROBLEM_NAME_ID; 
   typedef std::map<int, std::string> PROBLEM_ID_NAME; 
   
   template <class InputType> class IOHprofiler_suite : public IOHprofiler_problem<InputType>, public std::vector<std::shared_ptr<IOHprofiler_problem<InputType> > > {
   public:
     IOHprofiler_suite(std::vector<int> problem_id = std::vector<int>(0), std::vector<int> instance_id = std::vector<int>(0), std::vector<int> dimension = std::vector<int>(0)) : 
       suite_name("no suite"),
       problem_list_index(0),
       size_of_problem_list(0),
       get_problem_flag(false),
       load_problem_flag(false),
       current_problem(nullptr) {
       }
     ~IOHprofiler_suite() {}
   
     IOHprofiler_suite(const IOHprofiler_suite&) = delete;
     IOHprofiler_suite &operator =(const IOHprofiler_suite&) = delete;
   
     typedef std::shared_ptr<IOHprofiler_problem<InputType> > Problem_ptr;
     
     virtual void registerInSuite() {}
   
     void loadProblem();
   
     Problem_ptr get_next_problem();
     
     Problem_ptr get_current_problem();
     
   
     Problem_ptr get_problem(std::string problem_name, int instance, int dimension);
   
     Problem_ptr get_problem(int problem_id, int instance, int dimension);
   
     int IOHprofiler_suite_get_number_of_problems() const;
   
     int IOHprofiler_suite_get_number_of_instances() const;
   
     int IOHprofiler_suite_get_number_of_dimensions() const;
   
     std::vector<int> IOHprofiler_suite_get_problem_id() const;
   
     std::map<int, std::string> IOHprofiler_suite_get_problem_name() const;
   
     std::map<std::string, int> IOHprofiler_suite_get_problems() const;
   
     std::vector<int> IOHprofiler_suite_get_instance_id() const;
   
     std::vector<int> IOHprofiler_suite_get_dimension() const;
   
     std::string IOHprofiler_suite_get_suite_name() const;
   
     void IOHprofiler_set_suite_problem_id(const std::vector<int> &problem_id);
   
     void IOHprofiler_set_suite_instance_id(const std::vector<int> &instance_id);
   
     void IOHprofiler_set_suite_dimension(const std::vector<int> &dimension);
   
     void IOHprofiler_set_suite_name(const std::string suite_name);
   
     void mapIDTOName(const int id, const std::string name);
   
   private:
     std::string suite_name;
     int number_of_problems;
     int number_of_instances;
     int number_of_dimensions;
   
     std::vector<int> problem_id;
     std::vector<int> instance_id;
     std::vector<int> dimension;
   
     PROBLEM_ID_NAME problem_id_name_map;
     PROBLEM_NAME_ID problem_name_id_map;
   
     size_t problem_list_index;
     size_t size_of_problem_list;
     bool get_problem_flag;
     bool load_problem_flag;
   
     Problem_ptr current_problem;
   };
   
   #include "IOHprofiler_suite.hpp"
   
   #endif // _IOHPROFILER_SUITE_HPP
