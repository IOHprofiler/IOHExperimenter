
.. _program_listing_file_src_Template_Loggers_IOHprofiler_csv_logger.cpp:

Program Listing for File IOHprofiler_csv_logger.cpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_Loggers_IOHprofiler_csv_logger.cpp>` (``src/Template/Loggers/IOHprofiler_csv_logger.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "IOHprofiler_csv_logger.h"
   
   IOHprofiler_csv_logger::IOHprofiler_csv_logger() {
     this->output_directory = "./";
     this->folder_name = "IOHprofiler_test";
     this->algorithm_name =  "algorithm";
     this->algorithm_info = "algorithm_info";
   }
   
   IOHprofiler_csv_logger::IOHprofiler_csv_logger(std::string directory, std::string folder_name,
                           std::string alg_name, std::string alg_info) {
     this->output_directory = directory;
     this->folder_name = folder_name;
     this->algorithm_name =  alg_name;
     this->algorithm_info = alg_info;
   }
   
   IOHprofiler_csv_logger::~IOHprofiler_csv_logger() {
     this->clear_logger();
   }
   
   bool IOHprofiler_csv_logger::folder_exist(std::string folder_name) {
     // #if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
     //   DWORD ftyp = GetFileAttributesA(folder_name.c_str());
     //   if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
     //     return true;
     //   else
     //     return false;
     // #else
     //   std::fstream _file;
     //   _file.open(folder_name, std::ios::in);
     //   if(!_file) {
     //     return false;
     //   } else {
     //     return true;
     //   }
     // #endif
     
     // Check for existence.
     #if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
       if( _access(folder_name.c_str(), 0 ) != -1 ) {
         // Check for write permission.
         // Assume file is read-only.
         if( _access(folder_name.c_str(), 2 ) == -1 )
           return false;
         else 
           return true;
       } else {
         return false;
       }
     #else
       if( access(folder_name.c_str(), F_OK ) == 0) {
         if( access(folder_name.c_str(), W_OK ) == 0)
           return true;
         else
           return false;
       } else {
         return false;
       }
     #endif
   }
   
   void IOHprofiler_csv_logger::activate_logger() {
     openIndex();
   }
   
   int IOHprofiler_csv_logger::openIndex() { 
     std::string experiment_folder_name = IOHprofiler_experiment_folder_name();
     return IOHprofiler_create_folder(experiment_folder_name);
   }
   
   int IOHprofiler_csv_logger::IOHprofiler_create_folder(std::string folder_name) { 
   #if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)  
     if (mkdir(folder_name.c_str()) == 0) {
   #else
     if (mkdir(folder_name.c_str(),S_IRWXU) == 0) {
   #endif
       return 1;
     } else {
       IOH_error("Error on creating directory" + folder_name);
       return 0;
     }
   }
   
   std::string IOHprofiler_csv_logger::IOHprofiler_experiment_folder_name() {
     std::string renamed_directory = this->output_directory + IOHprofiler_path_separator + this->folder_name;
     std::string temp_folder_name = this->folder_name;
     int index = 0;
     //while (fs::exists(renamed_directory.c_str())) {
     while (folder_exist(renamed_directory) ) {
       ++index;
       temp_folder_name = this->folder_name  + '-' + _toString(index);
       renamed_directory = this->output_directory + IOHprofiler_path_separator + temp_folder_name;
     }
     this->folder_name = temp_folder_name;
     return renamed_directory;
   }
   
   void IOHprofiler_csv_logger::write_header() {
     std::string sub_directory_name = this->output_directory + IOHprofiler_path_separator 
                           + this->folder_name + IOHprofiler_path_separator
                           + "data_f" + _toString(problem_id)
                           + "_" + problem_name;
     
     std::string dat_header = "\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\"  \"best af(x)+b\"";
     if(this->logging_parameters.size() != 0) {
       for (std::map<std::string, std::shared_ptr<double> >::iterator iter = this->logging_parameters.begin(); iter != this->logging_parameters.end(); ++iter) {
         dat_header = dat_header + " \"" + iter->first + "\"";
       }
     }
   
     // if (!fs::exists(sub_directory_name.c_str())) {
     if (!folder_exist(sub_directory_name)) {
       IOHprofiler_create_folder(sub_directory_name.c_str());
     }
   
     if (complete_status()) {
       std::string cdat_name = sub_directory_name + IOHprofiler_path_separator 
                             + "IOHprofiler_f" + _toString(problem_id) + "_DIM"
                             + _toString(dimension) + ".cdat";
   
       if (this->cdat.is_open()) {
         this->cdat.close();
       }
       this->cdat.open(cdat_name.c_str(),std::ofstream::out | std::ofstream::app);
       //this->cdat << dat_header << '\n';
       this->cdat_buffer = "";
       this->write_in_buffer(dat_header+"\n",this->cdat_buffer,this->cdat);
     }
   
     if (interval_status()) {
       std::string idat_name = sub_directory_name + IOHprofiler_path_separator 
                             + "IOHprofiler_f" + _toString(problem_id) + "_DIM"
                             + _toString(dimension) + ".idat";
   
       if (this->idat.is_open()) {
         this->idat.close();
       }
       this->idat.open(idat_name.c_str(),std::ofstream::out | std::ofstream::app);
       //this->idat << dat_header << '\n';
       this->idat_buffer = "";
       this->write_in_buffer(dat_header+"\n",this->idat_buffer,this->idat);
     }
   
     if (update_status()) {
       std::string dat_name = sub_directory_name + IOHprofiler_path_separator 
                               + "IOHprofiler_f" + _toString(problem_id) + "_DIM"
                               + _toString(dimension) + ".dat";
   
       if (this->dat.is_open()) {
         this->dat.close();
       }
       this->dat_buffer = "";
       this->dat.open(dat_name.c_str(),std::ofstream::out | std::ofstream::app);
       //this->dat << dat_header << '\n';
       this->write_in_buffer(dat_header+"\n",this->dat_buffer,this->dat);
     }
   
     if (time_points_status()) {
       std::string tdat_name = sub_directory_name + IOHprofiler_path_separator 
                               + "IOHprofiler_f" + _toString(problem_id) + "_DIM"
                               + _toString(dimension) + ".tdat";
   
       if (this->tdat.is_open()) {
         this->tdat.close();
       }
       this->tdat_buffer = "";
       this->tdat.open(tdat_name.c_str(),std::ofstream::out | std::ofstream::app);
       //this->tdat << dat_header << '\n';
        this->write_in_buffer(dat_header+"\n",this->tdat_buffer,this->tdat);
     }
   }
   
   void IOHprofiler_csv_logger::track_problem(const int problem_id, const int dimension, const int instance, const std::string problem_name, const IOH_optimization_type maximization_minimization_flag){
     if (infoFile.is_open()) {
       write_info(this->instance, this->best_y[0], this->best_transformed_y[0], this->optimal_evaluations,
                                       this->last_y[0], this->last_transformed_y[0], this->last_evaluations);
     }
   
     this->optimal_evaluations = 0;
     this->last_evaluations = 0;
   
   
     if (maximization_minimization_flag == IOH_optimization_type::Maximization) {
       this->best_y.clear();
       this->best_y.push_back(-DBL_MAX);
       this->best_transformed_y.clear();
       this->best_transformed_y.push_back(-DBL_MAX);
       this->last_y.clear();
       this->last_y.push_back(-DBL_MAX);
       this->last_transformed_y.clear();
       this->last_transformed_y.push_back(-DBL_MAX);
     } else {
       this->best_y.clear();
       this->best_y.push_back(DBL_MAX);
       this->best_transformed_y.clear();
       this->best_transformed_y.push_back(DBL_MAX);
       this->last_y.clear();
       this->last_y.push_back(DBL_MAX);
       this->last_transformed_y.clear();
       this->last_transformed_y.push_back(DBL_MAX);
     }
     
     reset_observer(maximization_minimization_flag);
   
     this->problem_id = problem_id;
     this->dimension = dimension;
     this->instance = instance;
     this->problem_name = problem_name;
     this->maximization_minimization_flag = maximization_minimization_flag;
     
     openInfo(problem_id,dimension,problem_name);
     header_flag = false;
   }
   
   
   void IOHprofiler_csv_logger::track_problem(const IOHprofiler_problem<int> & problem) {
     // this->tracked_problem_int = nullptr;
     // this->tracked_problem_double = nullptr;
   
     this->track_problem(
       problem.IOHprofiler_get_problem_id(), 
       problem.IOHprofiler_get_number_of_variables(), 
       problem.IOHprofiler_get_instance_id(),
       problem.IOHprofiler_get_problem_name(),
       problem.IOHprofiler_get_optimization_type()
     );
     
     // this->problem_type = problem.IOHprofiler_get_problem_type();
     // this->tracked_problem_int = std::make_shared<IOHprofiler_problem<int> >(problem);
   }
   
   void IOHprofiler_csv_logger::track_problem(const IOHprofiler_problem<double> & problem) {
     // this->tracked_problem_int = nullptr;
     // this->tracked_problem_double = nullptr;
     this->track_problem(
       problem.IOHprofiler_get_problem_id(), 
       problem.IOHprofiler_get_number_of_variables(), 
       problem.IOHprofiler_get_instance_id(),
       problem.IOHprofiler_get_problem_name(),
       problem.IOHprofiler_get_optimization_type()
     );
     
     // this->problem_type = problem.IOHprofiler_get_problem_type();
     // this->tracked_problem_double = std::make_shared<IOHprofiler_problem<double> >(problem);
   }
   
   void IOHprofiler_csv_logger::track_suite(std::string suite_name){
     this->suite_name = suite_name;
   }
   
   void IOHprofiler_csv_logger::set_parameters(const std::vector<std::shared_ptr<double> > &parameters) {
     if (this->logging_parameters.size() != 0) {
       this->logging_parameters.clear();
     }
   
     for (size_t i = 0; i != parameters.size(); i++) {
       this->logging_parameters["parameter" + _toString(i+1)] = parameters[i];
     }
   }
   
   void IOHprofiler_csv_logger::set_parameters(const std::vector<std::shared_ptr<double> > &parameters, const std::vector<std::string> &parameters_name) {
     if (parameters_name.size() != parameters.size()) {
       IOH_error("Parameters and their names are given with different size.");
     }
     if (this->logging_parameters.size() != 0) {
       this->logging_parameters.clear();
     }
     for (size_t i = 0; i != parameters.size(); i++) {
       this->logging_parameters[parameters_name[i]] = parameters[i];
     }
   }
   
   
   void IOHprofiler_csv_logger::set_parameters_name(const std::vector<std::string > &parameters_name) {
     if (this->logging_parameters.size() != 0) {
       this->logging_parameters.clear();
     }
     //default value set as -9999.
     for (size_t i = 0; i != parameters_name.size(); i++) {
       this->logging_parameters[parameters_name[i]] = std::make_shared<double>(-9999);
     }
   }
   
   void IOHprofiler_csv_logger::set_parameters_name(const std::vector<std::string > &parameters_name, const std::vector<double> &initial_parameters) {
     if (parameters_name.size() != initial_parameters.size()) {
       IOH_error("Parameters and their names are given with different size.");
     }
     //default value set as -9999.
     for (size_t i = 0; i != parameters_name.size(); i++) {
       this->logging_parameters[parameters_name[i]] = std::make_shared<double>(initial_parameters[i]);
     }
   }
   
   void IOHprofiler_csv_logger::set_parameters(const std::vector<std::string > &parameters_name, const std::vector<double> &parameters) {
     if (parameters_name.size() != parameters.size()) {
       IOH_error("Parameters and their names are given with different size.");
     }
     for (size_t i = 0; i != parameters_name.size(); ++i)
     if(this->logging_parameters.find(parameters_name[i]) != this->logging_parameters.end()) {
       *this->logging_parameters[parameters_name[i]] = parameters[i];
     } else {
       IOH_error("Parameter " + parameters_name[i] + " does not exist");
     }
   }
   
   void IOHprofiler_csv_logger::add_dynamic_attribute(const std::vector<std::shared_ptr<double> > &attributes) {
     if (this->attr_per_run_name_value.size() != 0) {
       this->attr_per_run_name_value.clear();
     }
   
     for (size_t i = 0; i != attributes.size(); i++) {
       this->attr_per_run_name_value["attr" + _toString(i+1)] = attributes[i];
     }
   }
   
   void IOHprofiler_csv_logger::add_dynamic_attribute(const std::vector<std::shared_ptr<double> > &attributes, const std::vector<std::string> &attributes_name) {
     if (attributes_name.size() != attributes.size()) {
       IOH_error("Attributes and their names are given with different size.");
     }
     if (this->attr_per_run_name_value.size() != 0) {
       this->attr_per_run_name_value.clear();
     }
     for (size_t i = 0; i != attributes.size(); i++) {
       this->attr_per_run_name_value[attributes_name[i]] = attributes[i];
     }
   }
   
   void IOHprofiler_csv_logger::set_dynamic_attributes_name(const std::vector<std::string > &attributes_name) {
     if (this->attr_per_run_name_value.size() != 0) {
       this->attr_per_run_name_value.clear();
     }
     //default value set as -9999.
     for (size_t i = 0; i != attributes_name.size(); i++) {
       this->attr_per_run_name_value[attributes_name[i]] = std::make_shared<double>(-9999);
     }
   }
   
   void IOHprofiler_csv_logger::set_dynamic_attributes_name(const std::vector<std::string > &attributes_name, const std::vector<double> &initial_attributes) {
     if (attributes_name.size() != initial_attributes.size()) {
       IOH_error("Attributes and their names are given with different size.");
     }
     //default value set as -9999.
     for (size_t i = 0; i != attributes_name.size(); i++) {
       this->attr_per_run_name_value[attributes_name[i]] = std::make_shared<double>(initial_attributes[i]);
     }
   }
   
   void IOHprofiler_csv_logger::set_dynamic_attributes(const std::vector<std::string > &attributes_name, const std::vector<double> &attributes) {
     if (attributes_name.size() != attributes.size()) {
       IOH_error("Attributes and their names are given with different size.");
     }
     for (size_t i = 0; i != attributes_name.size(); ++i)
     if(this->attr_per_run_name_value.find(attributes_name[i]) != this->attr_per_run_name_value.end()) {
       *this->attr_per_run_name_value[attributes_name[i]] = attributes[i];
     } else {
       IOH_error("Dynamic attributes " + attributes_name[i] + " does not exist");
     }
   }
   
   void IOHprofiler_csv_logger::do_log(const std::vector<double> & log_info) {
     this->write_line( (size_t)(log_info[0]),log_info[1],log_info[2],log_info[3],log_info[4]);
   };
   
   void IOHprofiler_csv_logger::write_line(const std::vector<double> & log_info) {
     this->write_line( (size_t)(log_info[0]),log_info[1],log_info[2],log_info[3],log_info[4]);
   };
   
   void IOHprofiler_csv_logger::write_line(const size_t evaluations, const double y, const double best_so_far_y,
                    const double transformed_y, const double best_so_far_transformed_y) {
     if (header_flag == false) {
       this->write_header();
       header_flag = true;
     }
   
     this->last_evaluations = evaluations;
     this->last_y[0] = y;
     this->last_transformed_y[0] = transformed_y;
   
     bool cdat_flag = complete_trigger();
     bool idat_flag = interval_trigger(evaluations);
     bool dat_flag = update_trigger(transformed_y,maximization_minimization_flag);
     bool tdat_flag = time_points_trigger(evaluations);
   
     bool need_write =  cdat_flag || idat_flag || dat_flag || tdat_flag;
   
     if (need_write) {
       std::string written_line = _toString(evaluations) + " " + _toString(y) + " "
                                + _toString(best_so_far_y) + " " + _toString(transformed_y) + " "
                                + _toString(best_so_far_transformed_y);
       
       if (this->logging_parameters.size() != 0) {
         for (std::map<std::string, std::shared_ptr<double> >::iterator iter = this->logging_parameters.begin(); iter != this->logging_parameters.end(); ++iter) {
           written_line += " ";
           written_line += _toString(*(iter->second));
         }
       }
       
       written_line += '\n';
   
       if (cdat_flag) {
         if (!this->cdat.is_open()) {
           IOH_error("*.cdat file is not open");
         }
         //this->cdat << written_line;
         this->write_in_buffer(written_line,this->cdat_buffer,this->cdat);
       }
   
       if (idat_flag) {
         if (!this->idat.is_open()) {
           IOH_error("*.idat file is not open");
         }
         //this->idat << written_line;
         this->write_in_buffer(written_line,this->idat_buffer,this->idat);
       }
   
       if (dat_flag) {
         if (!this->dat.is_open()) {
           IOH_error("*.dat file is not open");
         }
         //this->dat << written_line;
         this->write_in_buffer(written_line,this->dat_buffer,this->dat);
       }
   
       if (tdat_flag) {
         if (!this->tdat.is_open()) {
           IOH_error("*.tdat file is not open");
         }
         //this->tdat << written_line;
         this->write_in_buffer(written_line,this->tdat_buffer,this->tdat);
       }
     }
   
     if (compareObjectives(transformed_y,this->best_transformed_y[0],this->maximization_minimization_flag)) {
       this->update_logger_info(evaluations,y,transformed_y);
     }
   }
   
   void IOHprofiler_csv_logger::openInfo(int problem_id, int dimension, std::string problem_name) {
     this->info_buffer = "";
     std::string titleflag = "";
     std::string optimization_type;
     if (this->maximization_minimization_flag == IOH_optimization_type::Maximization) {
       optimization_type = "T";
     } else {
       optimization_type = "F";
     }
   
     if (problem_id != this->last_problem_id) {
       this->infoFile.close();
       std::string infoFile_name = this->output_directory + IOHprofiler_path_separator 
                             + this->folder_name + IOHprofiler_path_separator
                             + "IOHprofiler_f" + _toString(problem_id)
                             + "_" + problem_name
                             + ".info";
       // if (fs::exists(infoFile_name.c_str())) {
       if (folder_exist(infoFile_name)) {
         titleflag = "\n";
       }
       this->infoFile.open(infoFile_name.c_str(),std::ofstream::out | std::ofstream::app);
       //this->infoFile << titleflag;
       this->info_buffer += titleflag;
       //this->infoFile << "suite = \"" << this->suite_name << "\", funcId = " <<  problem_id << ", funcName = \""<< problem_name << "\", DIM = "  << dimension << ", maximization = \"" << optimization_type << "\", algId = \"" << this->algorithm_name << "\", algInfo = \"" << this->algorithm_info << "\"";
         this->info_buffer += ("suite = \"" + this->suite_name + "\", funcId = " + _toString(problem_id) + ", funcName = \"" + problem_name + "\", DIM = "  + _toString(dimension) + ", maximization = \"" + optimization_type + "\", algId = \"" + this->algorithm_name + "\", algInfo = \"" + this->algorithm_info + "\"");
       if(this->attr_per_exp_name_value.size() != 0) {
         for (std::map<std::string,std::string>::iterator iter = this->attr_per_exp_name_value.begin(); iter != this->attr_per_exp_name_value.end(); iter++) {
               //this->infoFile << ", " << iter->first << " = \"" << iter->second << "\"";
           this->info_buffer += (", " + iter->first + " = \"" + iter->second + "\"");
         }
   
       }
       if (this->attr_per_run_name_value.size() != 0) {
         //this->infoFile << ", dynamicAttribute = \"";
         this->info_buffer +=  ", dynamicAttribute = \"";
         for (std::map<std::string,std::shared_ptr<double> >::iterator iter = this->attr_per_run_name_value.begin(); iter != this->attr_per_run_name_value.end();) {
               //this->infoFile << "|" << iter->first;
           this->info_buffer += (iter->first);
           if (++iter != this->attr_per_run_name_value.end()) {
             this->info_buffer += ("|");
           }
         }
         //this->infoFile << "\"";
         this->info_buffer += ("\"");
       }
       //this->infoFile << "\n%\n";
       this->info_buffer += ("\n%\n");
       //this->infoFile << "data_f" << problem_id << "_" << problem_name << "/IOHprofiler_f" << problem_id << "_DIM" << dimension << ".dat";     
       this->info_buffer += ("data_f" +  _toString(problem_id) + "_" + problem_name + "/IOHprofiler_f" + _toString(problem_id) + "_DIM" + _toString(dimension) + ".dat");     
       this->write_stream(this->info_buffer,this->infoFile);
       this->info_buffer.clear();
   
       this->last_problem_id = problem_id;
       this->last_dimension = dimension;
     } else if (dimension != this->last_dimension) {
       //this->infoFile << "\nsuite = \"" << this->suite_name << "\", funcId = " << problem_id << ", funcName = \""<< problem_name << "\", DIM = " << dimension << ", maximization = \"" << optimization_type << "\", algId = \"" << this->algorithm_name << "\", algInfo = \"" << this->algorithm_info << "\"";
       this->info_buffer += ("\nsuite = \"" + this->suite_name + "\", funcId = " + _toString(problem_id) + ", funcName = \"" + problem_name + "\", DIM = " + _toString(dimension) + ", maximization = \"" + optimization_type + "\", algId = \"" + this->algorithm_name + "\", algInfo = \"" + this->algorithm_info + "\"");
       if(this->attr_per_exp_name_value.size() != 0) {
         for (std::map<std::string,std::string>::iterator iter = this->attr_per_exp_name_value.begin(); iter != this->attr_per_exp_name_value.end(); iter++) {
               //this->infoFile << ", " << iter->first << " = \"" << iter->second << "\"";
           this->info_buffer += (", " + iter->first + " = \"" + iter->second + "\"");
         }
       }
   
       if (this->attr_per_run_name_value.size() != 0) {
         //this->infoFile << ", dynamicAttribute = \"";
         this->info_buffer += (", dynamicAttribute = \"");
         for (std::map<std::string,std::shared_ptr<double> >::iterator iter = this->attr_per_run_name_value.begin(); iter != this->attr_per_run_name_value.end();) {
               //this->infoFile << "|" << iter->first;
           this->info_buffer += (iter->first);
           if (++iter != this->attr_per_run_name_value.end()) {
             this->info_buffer += ("|");
           }
         }
         //this->infoFile << "\"";
         this->info_buffer += "\"";
       }
       //this->infoFile << "\n%\n";
       this->info_buffer += "\n%\n";
       //this->infoFile << "data_f" << problem_id << "_" << problem_name << "/IOHprofiler_f" << problem_id << "_DIM" << dimension << ".dat";    
       this->info_buffer += ("data_f" + _toString(problem_id) + "_" + problem_name + "/IOHprofiler_f" + _toString(problem_id) + "_DIM" + _toString(dimension) + ".dat");    
       this->write_stream(this->info_buffer,this->infoFile);
       this->info_buffer.clear();
   
       this->last_problem_id = problem_id;
       this->last_dimension = dimension;
     }
   }
   
   void IOHprofiler_csv_logger::write_info(int instance, double best_y, double best_transformed_y, int evaluations, 
                                           double last_y, double last_transformed_y, int last_evaluations) {
     if (!infoFile.is_open()) {
       IOH_error("write_info(): writing info into unopened infoFile");
     }
   
     this->info_buffer = "";
     //infoFile << ", " << instance << ":" << evaluations << "|" << best_y; 
     this->info_buffer += (", " + _toString(instance) + ":" + _toString(evaluations) + "|" + _toString(best_y));
     if(this->attr_per_run_name_value.size() != 0) {
       this->info_buffer += (";");
       for (std::map<std::string,std::shared_ptr<double> >::iterator iter = this->attr_per_run_name_value.begin(); iter != this->attr_per_run_name_value.end(); ) {
         //this->infoFile << "|" << *(iter->second);
         this->info_buffer += (_toString(*(iter->second)));
         if (++iter != this->attr_per_run_name_value.end()) {
           this->info_buffer += ("|");
         }
       }
     }
     this->write_stream(this->info_buffer,this->infoFile);
     this->info_buffer.clear();
   
     bool need_write = (evaluations != last_evaluations);
     if (need_write) {
       std::string written_line = _toString(last_evaluations) + " " + _toString(last_y) + " "
                                 + _toString(best_y) + " " + _toString(last_transformed_y) + " "
                                 + _toString(best_transformed_y);
       
       if (this->logging_parameters.size() != 0) {
         for (std::map<std::string, std::shared_ptr<double> >::iterator iter = this->logging_parameters.begin(); iter != this->logging_parameters.end(); ++iter) {
           written_line += " ";
           written_line += _toString(*(iter->second));
         }
       }
       
       written_line += '\n';
       if (this->cdat.is_open()) {
         //this->cdat << written_line;
         this->write_in_buffer(written_line,this->cdat_buffer,this->cdat);
         this->write_stream(this->cdat_buffer,this->cdat);
         this->cdat.flush();
         this->cdat_buffer.clear();
       }
       if (this->idat.is_open()) {
         //this->idat << written_line;
         this->write_in_buffer(written_line,this->idat_buffer,this->idat);
         this->write_stream(this->idat_buffer,this->idat);
         this->idat.flush();
         this->idat_buffer.clear();
       }
       if (this->dat.is_open()) {
         //this->dat << written_line;
         this->write_in_buffer(written_line,this->dat_buffer,this->dat);
         this->write_stream(this->dat_buffer,this->dat);
         this->dat.flush();
         this->dat_buffer.clear();
       }
       if (this->tdat.is_open()) {
         //this->tdat << written_line;
         this->write_in_buffer(written_line,this->tdat_buffer,this->tdat);
         this->write_stream(this->tdat_buffer,this->tdat);
         this->tdat.flush();
         this->tdat_buffer.clear();
       }
     }
   
     if (this->cdat.is_open()) {
         this->write_stream(this->cdat_buffer,this->cdat);
         this->cdat.flush();
         this->cdat_buffer.clear();
     }
     if (this->idat.is_open()) {
       this->write_stream(this->idat_buffer,this->idat);
       this->idat.flush();
       this->idat_buffer.clear();
     }
     if (this->dat.is_open()) {
       this->write_stream(this->dat_buffer,this->dat);
       this->dat.flush();
       this->dat_buffer.clear();
     }
     if (this->tdat.is_open()) {
       this->write_stream(this->tdat_buffer,this->tdat);
       this->tdat.flush();
       this->tdat_buffer.clear();
     }
   }
   
   void IOHprofiler_csv_logger::write_stream(const std::string buffer_string, std::fstream & dat_stream) {
     dat_stream.write(buffer_string.c_str(),sizeof(char)*buffer_string.size());
     //dat_stream << buffer_string;
   }
   
   
   void IOHprofiler_csv_logger::write_in_buffer(const std::string add_string, std::string & buffer_string, std::fstream & dat_stream) {
     if (buffer_string.size() + add_string.size() < MAX_BUFFER_SIZE) {
       buffer_string = buffer_string + add_string;
     } else {
       write_stream(buffer_string,dat_stream);
       buffer_string.clear();
       buffer_string = add_string;
     }
   }
   
   
   void IOHprofiler_csv_logger::update_logger_info(size_t optimal_evaluations, double y, double transformed_y) {
     this->optimal_evaluations = optimal_evaluations;
     this->best_y[0] =  y;
     this->best_transformed_y[0] = transformed_y;
   }
   
   
   void IOHprofiler_csv_logger::add_attribute(std::string name, double value) {
     this->attr_per_exp_name_value[name] = _toString(value);
   }
   
   void IOHprofiler_csv_logger::add_attribute(std::string name, int value) {
     this->attr_per_exp_name_value[name] = _toString(value);
   }
   
   void IOHprofiler_csv_logger::add_attribute(std::string name, float value) {
     this->attr_per_exp_name_value[name] = _toString(value);
   }
   
   void IOHprofiler_csv_logger::add_attribute(std::string name, std::string value) {
     this->attr_per_exp_name_value[name] = value;
   }
   
   void IOHprofiler_csv_logger::delete_attribute(std::string name) {
     this->attr_per_exp_name_value.erase(name);
   }
   
   void IOHprofiler_csv_logger::clear_logger() {
     if (infoFile.is_open()) {
       write_info(this->instance, this->best_y[0], this->best_transformed_y[0], this->optimal_evaluations,
                 this->last_y[0], this->last_transformed_y[0], this->last_evaluations);
       infoFile.close();
     }
   
     if (cdat.is_open()) {
       cdat.close();
     }
     if (idat.is_open()) {
       idat.close();
     }
     if (dat.is_open()) {
       dat.close();
     }
     if (tdat.is_open()) {
       tdat.close();
     }
   
     // if(this->tracked_problem_double != nullptr) {
     //   this->tracked_problem_double = nullptr;
     // }
   
     // if(this->tracked_problem_int != nullptr) {
     //   this->tracked_problem_int = nullptr;
     // }
   }
   
