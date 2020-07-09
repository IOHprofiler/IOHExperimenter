
.. _program_listing_file_src_Template_IOHprofiler_suite.hpp:

Program Listing for File IOHprofiler_suite.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_suite.hpp>` (``src/Template/IOHprofiler_suite.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   template <class InputType> void IOHprofiler_suite<InputType>::loadProblem() {
     if (this->size() != 0) {
       this->clear();
     }
     this->size_of_problem_list = this->number_of_dimensions * this->number_of_instances * this->number_of_problems;
     this->problem_list_index = 0;
   
     for (int i = 0; i != this->number_of_problems; ++i) {
       for (int j = 0; j != this->number_of_dimensions; ++j) {
         for (int h = 0; h != this->number_of_instances; ++h) {
           Problem_ptr p = get_problem(this->problem_id_name_map[this->problem_id[i]],
                                         this->instance_id[h],
                                         this->dimension[j]);
           this->push_back(p);
         }
       }
     }
     this->get_problem_flag = false;
     this->load_problem_flag = true;
   }
   
   template <class InputType> std::shared_ptr<IOHprofiler_problem<InputType> > IOHprofiler_suite<InputType>::get_next_problem() {
     if (this->load_problem_flag == false) {
       this->loadProblem();
     }
     
     if (this->size_of_problem_list == 0) {
       IOH_warning("There is no problem in the suite");
       return nullptr;
     }
   
     if (this->problem_list_index == this->size_of_problem_list - 1 && this->get_problem_flag == true) {
       return nullptr;
     }
   
     if (this->get_problem_flag == false) {
       this->get_problem_flag = true;
     } else {
       this->problem_list_index++;
     }    
     this->current_problem = (*this)[problem_list_index];
     
     this->current_problem->reset_problem();
     return this->current_problem;
   }
   
   template <class InputType> std::shared_ptr<IOHprofiler_problem<InputType> > IOHprofiler_suite<InputType>::get_current_problem() {
     if (this->load_problem_flag == false) {
       this->loadProblem();
     }
   
     if (this->get_problem_flag == false) {
       this->get_problem_flag = true;
     }
     this->current_problem = (*this)[this->problem_list_index];
     this->current_problem->reset_problem();
     return this->current_problem;
   }
   
   template <class InputType> std::shared_ptr<IOHprofiler_problem<InputType> > IOHprofiler_suite<InputType>::get_problem(std::string problem_name, int instance, int dimension) {
     Problem_ptr p = genericGenerator<IOHprofiler_problem<InputType> >::instance().create(problem_name);
     p->reset_problem();
     p->IOHprofiler_set_problem_id(this->problem_name_id_map[problem_name]);
     p->IOHprofiler_set_instance_id(instance);
     p->IOHprofiler_set_number_of_variables(dimension);
     return p;
   }
   
   template <class InputType> std::shared_ptr<IOHprofiler_problem<InputType> > IOHprofiler_suite<InputType>::get_problem(int problem_id, int instance, int dimension) {
     Problem_ptr p = genericGenerator<IOHprofiler_problem<InputType> >::instance().create(this->problem_id_name_map[problem_id]);
     p->reset_problem();
     p->IOHprofiler_set_problem_id(problem_id);
     p->IOHprofiler_set_instance_id(instance);
     p->IOHprofiler_set_number_of_variables(dimension);
     return p;
   }
   
   template <class InputType> int IOHprofiler_suite<InputType>::IOHprofiler_suite_get_number_of_problems() const {
     return this->number_of_problems;
   }
   
   template <class InputType> int IOHprofiler_suite<InputType>::IOHprofiler_suite_get_number_of_instances() const {
     return this->number_of_instances;
   }
   
   template <class InputType> int IOHprofiler_suite<InputType>::IOHprofiler_suite_get_number_of_dimensions() const {
     return this->number_of_dimensions;
   }
   
   template <class InputType> std::vector<int> IOHprofiler_suite<InputType>::IOHprofiler_suite_get_problem_id() const {
     return this->problem_id;
   }
   
   template <class InputType> std::map<int, std::string> IOHprofiler_suite<InputType>::IOHprofiler_suite_get_problem_name() const {
     return this->problem_id_name_map;
   }
   
   template <class InputType> std::map<std::string, int> IOHprofiler_suite<InputType>::IOHprofiler_suite_get_problems() const {
     return this->problem_name_id_map;
   }
   
   template <class InputType> std::vector<int> IOHprofiler_suite<InputType>::IOHprofiler_suite_get_instance_id() const {
     return this->instance_id;
   }
   
   template <class InputType> std::vector<int> IOHprofiler_suite<InputType>::IOHprofiler_suite_get_dimension() const {
     return this->dimension;
   }
   
   template <class InputType> std::string IOHprofiler_suite<InputType>::IOHprofiler_suite_get_suite_name() const {
     return this->suite_name;
   }
   
   template <class InputType> void IOHprofiler_suite<InputType>::IOHprofiler_set_suite_problem_id(const std::vector<int> &problem_id) {
     copyVector(problem_id,this->problem_id);
     this->number_of_problems = this->problem_id.size();
   }
   
   template <class InputType> void IOHprofiler_suite<InputType>::IOHprofiler_set_suite_instance_id(const std::vector<int> &instance_id) {
     copyVector(instance_id,this->instance_id);
     this->number_of_instances = this->instance_id.size();
   }
   
   template <class InputType> void IOHprofiler_suite<InputType>::IOHprofiler_set_suite_dimension(const std::vector<int> &dimension) {
     copyVector(dimension,this->dimension);
     this->number_of_dimensions = this->dimension.size();
   }
   
   template <class InputType> void IOHprofiler_suite<InputType>::IOHprofiler_set_suite_name(const std::string suite_name) {
     this->suite_name = suite_name;
   }
   
   template <class InputType> void IOHprofiler_suite<InputType>::mapIDTOName(const int id, const std::string name){
     problem_id_name_map[id] = name;
     problem_name_id_map[name] = id;
   }
