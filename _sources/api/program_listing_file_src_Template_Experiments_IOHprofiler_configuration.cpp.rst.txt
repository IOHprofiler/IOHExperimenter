
.. _program_listing_file_src_Template_Experiments_IOHprofiler_configuration.cpp:

Program Listing for File IOHprofiler_configuration.cpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_Experiments_IOHprofiler_configuration.cpp>` (``src/Template/Experiments/IOHprofiler_configuration.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "IOHprofiler_configuration.h"
   
   int IOHprofiler_configuration::set_Dict(_Dict &dict, const std::string section, const std::string key, const std::string value) {
     size_t i ;
   
     if (dict.n > 0) {
       for (i = 0; i < dict.n; ++i) {
         if (key.length() == 0) {
           continue;
         }
         if (value.length() == 0) {
           continue;
         }
         if (dict.key[i].length() == 0) {
           continue;
         }
         if (dict.section[i].length() == 0) {
           continue;
         }
         if ((key == dict.key[i]) && (section==dict.section[i])) {
           if (dict.value[i].length() != 0) {
             std::cout << "Multi setting of key(" << key << ") or section(" << section << ")\n";
             return -1;
           }
         }
       }
     }
     dict.section.push_back(strstrip(section));
     dict.key.push_back(strstrip(key));
     dict.value.push_back(strstrip(value));
     dict.n += 1;
     return 0 ;
   }
   
   std::string IOHprofiler_configuration::get_Dict_String(const _Dict dict, const std::string section, const std::string key){
     size_t i;
     if (key.length() == 0) {
       std::cout << "EMPTY KEY INPUT.\n";
     }
     if (section.length() == 0) {
       std::cout << "EMPTY SECTION INPUT.\n";
     }
     for (i = 0; i < dict.n; ++i) {
       if (dict.key[i].length() == 0) {
         continue;
       }
       if (dict.section[i].length() == 0) {
         continue;
       }
       if ((key == dict.key[i]) && (section == dict.section[i])) {
         return dict.value[i];
       }
     }
     std::cout << "Can not find the corresponding configuration for key: " << key << " in section : "<< section << "\n";
     return NULL;
   }
   
   std::vector<int> IOHprofiler_configuration::get_Dict_int_vector(const _Dict dict, const std::string section, const std::string key, const int _min, const int _max){
     size_t i;
     std::vector<int> result;
     if (key.length() == 0) {
       std::cout << "EMPTY KEY INPUT.\n";
     }
     if (section.length() == 0){
       std::cout << "EMPTY SECTION INPUT.\n";
     }
     for (i = 0; i < dict.n; ++i) {
       if (dict.key[i].length() == 0) {
         continue;
       }
       if (dict.section[i].length() == 0) {
         continue;
       }
       if (key == dict.key[i] && section == dict.section[i]) {
         result = get_int_vector_parse_string(dict.value[i],_min,_max);
         return result;
       }
     }
     std::cout << "Can not find the corresponding configuration for key: "<< key << " in section : "<< section << "\n";
     return result;
   }
   
   int IOHprofiler_configuration::get_Dict_Int(const _Dict dict, const std::string section, const std::string key){
     int result;
     std::string str;
     str = get_Dict_String(dict, section, key);
     result = std::stoi(str);
     return result;
   }
   
   bool IOHprofiler_configuration::get_Dict_bool(const _Dict dict, const std::string section, const std::string key){
     bool result = false;
     std::string str;
     str = get_Dict_String(dict, section, key);
     transform(str.begin(), str.end(), str.begin(), ::tolower);
     if (str == "true") {
       result = true;
     }
     return result;
   }
   
   linecontent IOHprofiler_configuration::add_Line(const std::string input_line, std::string &section, std::string &key, std::string &value){
     linecontent content;
     std::string line;
     size_t len;
     char tempkey[MAXKEYNUMBER];
     char tempvalue[MAXKEYNUMBER];
     char tempsection[MAXKEYNUMBER];
   
     line = strstrip(input_line);
     len = line.length();
   
     if (len < 0) {
       content = EMPTY;
     } else if (line[0] == '#' || line[0] == ';') {
       content = COMMENT;
     } else if(line[0] == '['  && line[len-1] == ']') {
       sscanf(line.c_str(), "[%[^]]",tempsection);
       section = tempsection;
       content = SECTION;
     } else if(sscanf (line.c_str(), "%[^=] = \"%[^\"]", tempkey, tempvalue) == 2 || sscanf (line.c_str(), "%[^=] = '%[^\']", tempkey, tempvalue) == 2) {
       value = tempvalue;
       key = tempkey;
       content = VALUE;
     } else if(sscanf (line.c_str(), "%[^=] = %[^;#]", tempkey, tempvalue) == 2) {
       value = tempvalue;
       key = tempkey;
       content = VALUE;
     } else {
       content = CON_ERROR;
     }
     return content;
   }
   
   _Dict IOHprofiler_configuration::load(const std::string filename) {
     std::ifstream fp(filename.c_str());
   
     std::string line;
     std::string section;
     std::string key;
     std::string value;
   
     int len;
     _Dict dict;
     linecontent lc;
   
     if (!fp.is_open()) {
       std::cout << "Cannot open file " << filename << std::endl;
     }
   
     while (!fp.eof()) {
       getline(fp,line);
       len = line.length() -1;
       if (len <= 0) {
         continue;
       }
   
       lc = add_Line(line,section,key,value);
       switch(lc) {
         case EMPTY:
           break;
         case COMMENT:
           break;
         case SECTION:
           strstrip(section);
           this->set_Dict(dict,section,key,value);
           break;
         case VALUE:
           strstrip(key);
           this->set_Dict(dict,section,key,value);
           break;
         case CON_ERROR:
           std::cout << "There is an error for line: \" "<< line << "\"";
           break;
       }
       if (lc == CON_ERROR) {
         break;
       }
     }
     fp.close();
     return dict;
   }
   
   void IOHprofiler_configuration::readcfg(std::string filename){
     _Dict dict;
     dict = load(filename);
     suite_name = get_Dict_String(dict,"suite","suite_name");
     if (suite_name == "BBOB") {
       max_number_of_problem = 24;
       max_dimension = 100;
     } else if (suite_name == "PBO") {
       max_dimension = 20000;
       max_number_of_problem = 23;
     }
     problem_id = get_Dict_int_vector(dict,"suite","problem_id",1,max_number_of_problem);
     instance_id = get_Dict_int_vector(dict,"suite","instance_id",1,100);
     dimension = get_Dict_int_vector(dict,"suite","dimension",1,max_dimension);
   
     output_directory = get_Dict_String(dict,"logger","output_directory");
     result_folder = get_Dict_String(dict,"logger","result_folder");
     algorithm_info = get_Dict_String(dict,"logger","algorithm_info");
     algorithm_name = get_Dict_String(dict,"logger","algorithm_name");
   
     complete_triggers = get_Dict_bool(dict,"observer","complete_triggers");
     update_triggers = get_Dict_bool(dict,"observer","update_triggers");
     base_evaluation_triggers = get_Dict_int_vector(dict,"observer","base_evaluation_triggers",0,10);
     number_target_triggers = get_Dict_Int(dict,"observer","number_target_triggers");
     number_interval_triggers = get_Dict_Int(dict,"observer","number_interval_triggers");
     }
   
   std::string IOHprofiler_configuration::get_suite_name() {
     return this->suite_name;
   }
   
   std::vector<int> IOHprofiler_configuration::get_problem_id() {
     return this->problem_id;
   }
   
   std::vector<int> IOHprofiler_configuration::get_instance_id() {
     return this->instance_id;
   }
   
   std::vector<int> IOHprofiler_configuration::get_dimension() {
     return this->dimension;
   }
   
   std::string IOHprofiler_configuration::get_output_directory() {
     return this->output_directory;
   }
   
   std::string IOHprofiler_configuration::get_result_folder() {
     return this->result_folder;
   }
   
   std::string IOHprofiler_configuration::get_algorithm_info() {
     return this->algorithm_info;
   }
   
   std::string IOHprofiler_configuration::get_algorithm_name() {
     return this->algorithm_name;
   }
   
   bool IOHprofiler_configuration::get_complete_triggers() {
     return this->complete_triggers;
   }
   
   bool IOHprofiler_configuration::get_update_triggers() {
     return this->update_triggers;
   }
   
   std::vector<int> IOHprofiler_configuration::get_base_evaluation_triggers() {
     return this->base_evaluation_triggers;
   }
   
   int IOHprofiler_configuration::get_number_target_triggers() {
     return this->number_target_triggers;
   }
   
   int IOHprofiler_configuration::get_number_interval_triggers() {
     return this->number_interval_triggers;
   }
