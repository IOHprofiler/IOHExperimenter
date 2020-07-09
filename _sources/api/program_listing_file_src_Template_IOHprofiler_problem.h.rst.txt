
.. _program_listing_file_src_Template_IOHprofiler_problem.h:

Program Listing for File IOHprofiler_problem.h
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_problem.h>` (``src/Template/IOHprofiler_problem.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_PROBLEM_H
   #define _IOHPROFILER_PROBLEM_H
   
   #include "IOHprofiler_common.h"
   #include "IOHprofiler_transformation.h"
   
   static IOHprofiler_transformation transformation;
   
   template <class InputType> class IOHprofiler_problem 
   {
   public:
     IOHprofiler_problem(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) : 
       problem_id(DEFAULT_PROBLEM_ID), 
       instance_id(instance_id), 
       maximization_minimization_flag(IOH_optimization_type::Maximization),
       number_of_variables(DEFAULT_DIMENSION), 
       number_of_objectives(1),
       lowerbound(std::vector<InputType> (number_of_variables) ), 
       upperbound(std::vector<InputType> (number_of_variables) ),
       optimal(std::vector<double>(number_of_objectives) ),
       optimalFound(false),
       evaluations(0),
       best_so_far_raw_objectives(std::vector<double>(number_of_objectives) ),
       best_so_far_raw_evaluations(0),
       best_so_far_transformed_objectives(std::vector<double>(number_of_objectives) ) {}
     
     virtual ~IOHprofiler_problem() {}
   
     IOHprofiler_problem(const IOHprofiler_problem&) = delete;
     IOHprofiler_problem &operator=(const IOHprofiler_problem&) = delete;
     
     // virtual std::vector<double> internal_evaluate_multi (const std::vector<InputType> &x) {
     //   std::vector<double> result;
     //   std::cout << "No multi evaluate function defined" << std::endl;
     //   return result;
     // };
   
     virtual double internal_evaluate(const std::vector<InputType> &x) {
       double result = std::numeric_limits<double>::lowest();
       IOH_warning("No evaluate function defined");
       return result;
     }
   
     virtual void prepare_problem() {
     }
   
     // std::vector<double> evaluate_multi(std::vector<InputType> x) {
     //   ++this->evaluations;
   
     //   transformation.variables_transformation(x,this->problem_id,this->instance_id,this->problem_type);
     //   this->raw_objectives = internal_evaluate_multi(x);
       
     //   this->transformed_objectives = this->raw_objectives;
     //   transformation.objectives_transformation(x,this->transformed_objectives,this->problem_id,this->instance_id,this->problem_type);
     //   if (compareObjectives(this->transformed_objectives,this->best_so_far_transformed_objectives,this->maximization_minimization_flag)) {
     //     this->best_so_far_transformed_objectives = this->transformed_objectives;
     //     this->best_so_far_transformed_evaluations = this->evaluations;
     //     this->best_so_far_raw_objectives = this->raw_objectives;
     //     this->best_so_far_raw_evaluations = this->evaluations;
       
     //   }
   
     //   if (compareVector(this->transformed_objectives,this->optimal)) {
     //     this->optimalFound = true;
     //   }
   
     //   return this->transformed_objectives;
     // }
   
     double evaluate(std::vector<InputType> x);
   
     virtual void customize_optimal(){
   
     }
   
     void calc_optimal();
   
     // virtual std::vector<double> constraints() {
     //   std::vector<double> con;
     //   printf("No constraints function defined\n");
     //   return con;
     // };
     
     void reset_problem();
   
     std::vector<double> loggerCOCOInfo() const;
     
     std::vector<double> loggerInfo() const;
     
     bool IOHprofiler_hit_optimal() const;
   
     int IOHprofiler_get_problem_id() const;
   
     void IOHprofiler_set_problem_id(int problem_id);
     
     int IOHprofiler_get_instance_id() const;
     
     void IOHprofiler_set_instance_id(int instance_id);
   
     std::string IOHprofiler_get_problem_name() const;
   
     void IOHprofiler_set_problem_name(std::string problem_name);
   
     std::string IOHprofiler_get_problem_type() const;
   
     void IOHprofiler_set_problem_type(std::string problem_type);
   
     std::vector<InputType> IOHprofiler_get_lowerbound() const;
     
     void IOHprofiler_set_lowerbound(int lowerbound);
   
     void IOHprofiler_set_lowerbound(const std::vector<InputType> &lowerbound);
   
     std::vector<InputType> IOHprofiler_get_upperbound() const;
   
     void IOHprofiler_set_upperbound(int upperbound);
   
     void IOHprofiler_set_upperbound(const std::vector<InputType> &upperbound);
    
     int IOHprofiler_get_number_of_variables() const;
   
     void IOHprofiler_set_number_of_variables(const int number_of_variables);
   
     void IOHprofiler_set_number_of_variables(const int number_of_variables, const std::vector<InputType> &best_variables);
   
     int IOHprofiler_get_number_of_objectives() const;
   
     void IOHprofiler_set_number_of_objectives(const int number_of_objectives);
   
     std::vector<double> IOHprofiler_get_raw_objectives() const;
   
     std::vector<double> IOHprofiler_get_transformed_objectives() const;
   
     int IOHprofiler_get_transformed_number_of_variables() const;
   
     std::vector<InputType> IOHprofiler_get_transformed_variables() const;
   
     std::vector<InputType> IOHprofiler_get_best_variables() const;
   
     void IOHprofiler_set_best_variables(const InputType best_variables);
   
     void IOHprofiler_set_best_variables(const std::vector<InputType> &best_variables);
   
     std::vector<double> IOHprofiler_get_optimal() const;
   
     void IOHprofiler_set_optimal(const double optimal);
   
     void IOHprofiler_set_optimal(const std::vector<double> &optimal);
   
     void IOHprofiler_evaluate_optimal(std::vector<InputType> best_variables);
   
     void IOHprofiler_evaluate_optimal();
   
     int IOHprofiler_get_evaluations() const;
   
     std::vector<double> IOHprofiler_get_best_so_far_raw_objectives() const;
   
     int IOHprofiler_get_best_so_far_raw_evaluations() const;
   
     std::vector<double> IOHprofiler_get_best_so_far_transformed_objectives() const;
   
     int IOHprofiler_get_best_so_far_transformed_evaluations() const;
   
     IOH_optimization_type IOHprofiler_get_optimization_type() const;
   
     void IOHprofiler_set_as_maximization();
   
     void IOHprofiler_set_as_minimization();
   
   private:
     int problem_id; 
     int instance_id; 
     
     std::string problem_name;
     std::string problem_type;   
   
     IOH_optimization_type maximization_minimization_flag;
   
     std::size_t number_of_variables; 
     std::size_t number_of_objectives;
   
     std::vector<InputType> lowerbound;
     std::vector<InputType> upperbound;
   
     std::vector<InputType> best_variables; 
     std::vector<InputType> best_transformed_variables;
     std::vector<double> optimal; 
     bool optimalFound;
   
     std::vector<double> raw_objectives; 
     std::vector<double> transformed_objectives; 
     int transformed_number_of_variables; 
     std::vector<InputType> transformed_variables; 
   
   
     std::size_t evaluations; 
     std::vector<double> best_so_far_raw_objectives; 
     int best_so_far_raw_evaluations; 
     std::vector<double> best_so_far_transformed_objectives; 
     int best_so_far_transformed_evaluations; 
   };
   
   #include "IOHprofiler_problem.hpp"
   
   #endif // _IOHPROFILER_PROBLEM_H
