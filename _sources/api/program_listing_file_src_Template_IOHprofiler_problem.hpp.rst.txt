
.. _program_listing_file_src_Template_IOHprofiler_problem.hpp:

Program Listing for File IOHprofiler_problem.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_problem.hpp>` (``src/Template/IOHprofiler_problem.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   template <class InputType> double IOHprofiler_problem<InputType>::evaluate(std::vector<InputType> x) {
     ++this->evaluations;
   
     if(x.size() != this->number_of_variables) {
       IOH_warning("The dimension of solution is incorrect.");
       if (this->maximization_minimization_flag == IOH_optimization_type::Maximization) {
         this->raw_objectives[0] = std::numeric_limits<double>::lowest();
         this->transformed_objectives[0] = std::numeric_limits<double>::lowest();
       } else {
         this->raw_objectives[0] = std::numeric_limits<double>::max();
         this->transformed_objectives[0] = std::numeric_limits<double>::max();
       }
       return this->transformed_objectives[0];
     }
   
     transformation.variables_transformation(x,this->problem_id,this->instance_id,this->problem_type);
     this->raw_objectives[0] = internal_evaluate(x);
     
     this->transformed_objectives[0] = this->raw_objectives[0];
   
     transformation.objectives_transformation(x,this->transformed_objectives,this->problem_id,this->instance_id,this->problem_type);
     if (compareObjectives(this->transformed_objectives,this->best_so_far_transformed_objectives,this->maximization_minimization_flag)) {
       this->best_so_far_transformed_objectives = this->transformed_objectives;
       this->best_so_far_transformed_evaluations = this->evaluations;
       this->best_so_far_raw_objectives = this->raw_objectives;
       this->best_so_far_raw_evaluations = this->evaluations;
     }
   
     if (compareVector(this->transformed_objectives,this->optimal)) {
       this->optimalFound = true;
     }
   
     return this->transformed_objectives[0];
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::calc_optimal() {
     if (this->best_variables.size() == this->number_of_variables) {
       if (this->number_of_objectives == 1) {
         if(this->problem_type == "bbob") {
           Coco_Transformation_Data::raw_x.clear();
           for (int i = 0; i != this->best_variables.size(); ++i) {
             Coco_Transformation_Data::raw_x.push_back(this->best_variables[i]);
           }
         }
         this->optimal[0] = internal_evaluate(this->best_variables);
       } else {
         IOH_warning("Multi-objectives optimization is not supported now.");
       }
       transformation.objectives_transformation(this->best_variables,this->optimal,this->problem_id,this->instance_id,this->problem_type);
     }
     else {
       this->optimal.clear();
       for (int i = 0; i < this->number_of_objectives; ++i) {
         if (this->maximization_minimization_flag == IOH_optimization_type::Maximization) {
           this->optimal.push_back(std::numeric_limits<double>::max()); 
         } else {
           this->optimal.push_back(std::numeric_limits<double>::lowest());
         }
       }
       customize_optimal();
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::reset_problem() {
     this->evaluations = 0;
     this->best_so_far_raw_evaluations = 0;
     this->best_so_far_transformed_evaluations = 0;
     this->optimalFound = false;
     for (int i = 0; i !=  this->number_of_objectives; ++i) {
       if (this->maximization_minimization_flag == IOH_optimization_type::Maximization) {
         this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::lowest();
         this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::lowest();
       } else {
         this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::max();
         this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::max();
       }
     }
     this->prepare_problem();
     this->calc_optimal();
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::loggerCOCOInfo() const{
     std::vector<double> logger_info(5);
     logger_info[0] = (double)this->evaluations;
     logger_info[1] = this->transformed_objectives[0] - this->optimal[0];
     logger_info[2] = this->best_so_far_transformed_objectives[0] - this->optimal[0];
     logger_info[3] = this->transformed_objectives[0];
     logger_info[4] = this->best_so_far_transformed_objectives[0];
   
     return logger_info;
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::loggerInfo() const{
     std::vector<double> logger_info(5);
     logger_info[0] = (double)this->evaluations;
     logger_info[1] = this->raw_objectives[0];
     logger_info[2] = this->best_so_far_raw_objectives[0];
     logger_info[3] = this->transformed_objectives[0];
     logger_info[4] = this->best_so_far_transformed_objectives[0];
     return logger_info;
   }
   
   template <class InputType> bool IOHprofiler_problem<InputType>::IOHprofiler_hit_optimal() const {
     return this->optimalFound;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_problem_id() const {
     return this->problem_id;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_problem_id(int problem_id){
     this->problem_id = problem_id;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_instance_id() const {
     return this->instance_id;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_instance_id(int instance_id) {
     this->instance_id = instance_id;
     this->prepare_problem();
     this->calc_optimal();
   }
   
   template <class InputType> std::string IOHprofiler_problem<InputType>::IOHprofiler_get_problem_name() const {
     return this->problem_name;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_problem_name(std::string problem_name) {
     this->problem_name = problem_name;
   }
   
   template <class InputType> std::string IOHprofiler_problem<InputType>::IOHprofiler_get_problem_type() const {
     return this->problem_type;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_problem_type(std::string problem_type) {
     this->problem_type = problem_type;
   }
   
   template <class InputType> std::vector<InputType> IOHprofiler_problem<InputType>::IOHprofiler_get_lowerbound() const {
     return this->lowerbound;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_lowerbound(int lowerbound) {
     std::vector<InputType>().swap(this->lowerbound);
     this->lowerbound.reserve(this->number_of_variables);
     for (int i = 0; i < this->number_of_variables; ++i) {
       this->lowerbound.push_back(lowerbound);
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_lowerbound(const std::vector<InputType> &lowerbound) {
     this->lowerbound = lowerbound;
   }
   
   template <class InputType> std::vector<InputType> IOHprofiler_problem<InputType>::IOHprofiler_get_upperbound() const {
     return this->upperbound;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_upperbound(int upperbound) {
     std::vector<InputType>().swap(this->upperbound);
     this->upperbound.reserve(this->number_of_variables);
     for (int i = 0; i < this->number_of_variables; ++i) {
       this->upperbound.push_back(upperbound);
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_upperbound(const std::vector<InputType> &upperbound) {
     this->upperbound = upperbound;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_number_of_variables() const {
     return this->number_of_variables;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_number_of_variables(const int number_of_variables) {
     this->number_of_variables = number_of_variables;
     if (this->best_variables.size() != 0) {
       this->IOHprofiler_set_best_variables(this->best_variables[0]);
     }
     if (this->lowerbound.size() != 0) {
       this->IOHprofiler_set_lowerbound(this->lowerbound[0]);
     }
     if (this->upperbound.size() != 0) {
       this->IOHprofiler_set_upperbound(this->upperbound[0]);
     }
     this->prepare_problem();
     this->calc_optimal();
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_number_of_variables(const int number_of_variables, const std::vector<InputType> &best_variables) {
     this->number_of_variables = number_of_variables;
     this->best_variables = best_variables;
     if (this->lowerbound.size() != 0) {
       this->IOHprofiler_set_lowerbound(this->lowerbound[0]);
     }
     if (this->upperbound.size() != 0) {
       this->IOHprofiler_set_upperbound(this->upperbound[0]);
     }
     this->prepare_problem();
     this->calc_optimal();
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_number_of_objectives() const {
     return this->number_of_objectives;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_number_of_objectives(const int number_of_objectives) {
     this->number_of_objectives = number_of_objectives;
     this->raw_objectives = std::vector<double>(this->number_of_objectives);
     this->transformed_objectives = std::vector<double>(this->number_of_objectives);
     if (this->maximization_minimization_flag == IOH_optimization_type::Maximization) {
       this->best_so_far_raw_objectives = std::vector<double>(this->number_of_objectives,std::numeric_limits<double>::lowest());
       this->best_so_far_transformed_objectives = std::vector<double>(this->number_of_objectives,std::numeric_limits<double>::lowest());
     } else {
       this->best_so_far_raw_objectives = std::vector<double>(this->number_of_objectives,std::numeric_limits<double>::max());
       this->best_so_far_transformed_objectives = std::vector<double>(this->number_of_objectives,std::numeric_limits<double>::max());
     }
     this->optimal = std::vector<double>(this->number_of_objectives);
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::IOHprofiler_get_raw_objectives() const {
     return this->raw_objectives;
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::IOHprofiler_get_transformed_objectives() const {
     return this->transformed_objectives;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_transformed_number_of_variables() const {
     return this->transformed_number_of_variables;
   }
   
   template <class InputType> std::vector<InputType> IOHprofiler_problem<InputType>::IOHprofiler_get_transformed_variables() const {
     return this->transformed_variables;
   }
   
   template <class InputType> std::vector<InputType> IOHprofiler_problem<InputType>::IOHprofiler_get_best_variables() const {
     return this->best_variables;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_best_variables(const InputType best_variables) {
     this->best_variables.clear();
     for (int i = 0; i < this->number_of_variables; ++i) {
       this->best_variables.push_back(best_variables);
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_best_variables(const std::vector<InputType> &best_variables) {
     this->best_variables = best_variables;
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::IOHprofiler_get_optimal() const {
     return this->optimal;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_optimal(const double optimal) {
     std::vector<double>().swap(this->optimal);
     this->optimal.reserve(this->number_of_objectives);
     for (int i = 0; i < this->number_of_objectives; ++i) {
       this->optimal.push_back(optimal);
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_optimal(const std::vector<double> &optimal) {
     this->optimal = optimal;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_evaluate_optimal(std::vector<InputType> best_variables) {
     this->optimal[0] = this->evaluate(best_variables);
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_evaluate_optimal() {
     this->optimal[0] = this->evaluate(this->best_variables);
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_evaluations() const {
     return this->evaluations;
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::IOHprofiler_get_best_so_far_raw_objectives() const {
     return this->best_so_far_raw_objectives;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_best_so_far_raw_evaluations() const {
     return this-> best_so_far_raw_evaluations;
   }
   
   template <class InputType> std::vector<double> IOHprofiler_problem<InputType>::IOHprofiler_get_best_so_far_transformed_objectives() const {
     return this->best_so_far_transformed_objectives;
   }
   
   template <class InputType> int IOHprofiler_problem<InputType>::IOHprofiler_get_best_so_far_transformed_evaluations() const {
     return this->best_so_far_transformed_evaluations;
   }
   
   template <class InputType> IOH_optimization_type IOHprofiler_problem<InputType>::IOHprofiler_get_optimization_type() const {
     return this->maximization_minimization_flag;
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_as_maximization() {
     this->maximization_minimization_flag = IOH_optimization_type::Maximization;
     for (int i = 0; i !=  this->number_of_objectives; ++i) {
       this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::lowest();
       this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::lowest();
     }
   }
   
   template <class InputType> void IOHprofiler_problem<InputType>::IOHprofiler_set_as_minimization() {
     this->maximization_minimization_flag = IOH_optimization_type::Minimization;
     for (int i = 0; i !=  this->number_of_objectives; ++i) {
       this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::max();
       this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::max();
     }
   }
