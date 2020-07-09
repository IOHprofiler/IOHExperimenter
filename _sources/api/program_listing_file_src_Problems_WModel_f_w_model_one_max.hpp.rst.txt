
.. _program_listing_file_src_Problems_WModel_f_w_model_one_max.hpp:

Program Listing for File f_w_model_one_max.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_WModel_f_w_model_one_max.hpp>` (``src/Problems/WModel/f_w_model_one_max.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_W_MODEL_ONE_MAX_HPP
   #define _F_W_MODEL_ONE_MAX_HPP
   
   #include "IOHprofiler_problem.h"
   #include "wmodels.hpp"
   
   class W_Model_OneMax : public IOHprofiler_problem<int> {
   public:
     W_Model_OneMax(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("W_Model_OneMax");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~W_Model_OneMax() {};
   
     std::vector<int> dummy_info;
     double dummy_para = 0;
     int epistasis_para = 0;
     int neutrality_para = 0;
     int ruggedness_para = 0;
     std::vector<int> ruggedness_info;
     int temp_dimension;
     void prepare_problem() {
       this->temp_dimension = this->IOHprofiler_get_number_of_variables();
   
       if (this->dummy_para > 0) {
         this->dummy_info = dummy(this->temp_dimension,dummy_para,10000);
         assert(this->dummy_info.size()==(size_t)(this->temp_dimension*this->dummy_para));
         this->temp_dimension = this->dummy_info.size();
       }
   
       if (this->neutrality_para > 0) {
         this->temp_dimension = this->temp_dimension / this->neutrality_para;
       }
       if (this->ruggedness_para > 0) {
         this->ruggedness_info = ruggedness_raw(ruggedness_translate(this->ruggedness_para, this->temp_dimension), this->temp_dimension); 
       }
     }
   
     void customize_optimal() {
       int dimension = IOHprofiler_get_number_of_variables();
       int optimal_value = (int)(dimension*(this->dummy_para == 0 ? 1 : this->dummy_para)) / (this->neutrality_para == 0 ? 1 : this->neutrality_para);
   
       IOHprofiler_set_optimal((double)optimal_value);
     }
     
     void set_w_setting(const double dummy_para, const int epistasis_para, const int neutrality_para, const int ruggedness_para) {
       this->dummy_para = dummy_para;
       this->epistasis_para = epistasis_para;
       this->neutrality_para = neutrality_para;
       this->ruggedness_para = ruggedness_para;
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       std::vector<int> w_model_x;
       std::vector<int> tempX;
       int n;
       
       if (this->dummy_para > 0){
         n = this->dummy_info.size();
         w_model_x.reserve(n);
         for (int i = 0; i != n; ++i) {
           w_model_x.push_back(x[this->dummy_info[i]]);
         }
       } else {
         w_model_x = x;
       }
       
       // neutrality layer
       if (this->neutrality_para > 0) {
         tempX = w_model_x;
         layer_neutrality_compute(tempX,w_model_x,this->neutrality_para);
       } 
       
       // epistasis layer
       if (this->epistasis_para > 0) {
         tempX = w_model_x;
         layer_epistasis_compute(tempX,w_model_x,this->epistasis_para);
       }
       
       // base evaluate
       n = w_model_x.size();
       int result = 0;
       for (int i = 0; i != n; ++i) {
         result += w_model_x[i];
       }
   
       // ruggedness layer
       if (this->ruggedness_para > 0) {
         result = this->ruggedness_info[result];
       }
       
       return (double)result;
     }
   
     static W_Model_OneMax * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new W_Model_OneMax(instance_id, dimension);
     }
   };
   
   #endif
