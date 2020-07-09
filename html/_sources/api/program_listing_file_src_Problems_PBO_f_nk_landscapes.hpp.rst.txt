
.. _program_listing_file_src_Problems_PBO_f_nk_landscapes.hpp:

Program Listing for File f_nk_landscapes.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_nk_landscapes.hpp>` (``src/Problems/PBO/f_nk_landscapes.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_NK_LANDSCAPES_H
   #define _F_NK_LANDSCAPES_H
   
   #include "IOHprofiler_problem.h"
   
   class NK_Landscapes : public IOHprofiler_random, public IOHprofiler_problem<int> {
   public:
     NK_Landscapes(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("NK_Landscapes");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
     
     ~NK_Landscapes() {};
     
     std::vector<std::vector<double>> F;
     std::vector<std::vector<int>> E;
     int k = 1;
   
     void set_n_k(const int n, const int k) {
       E.clear();
       F.clear();
       this->k = k;
       if(k > n) {
         IOH_error("NK_Landscapes, k > n");
       }
       std::vector<double> rand_vec;
       for (int i = 0;i != n; ++i) {
           IOHprofiler_uniform_rand((size_t)k,(long)(k * (i+1)),rand_vec);
   
   
           std::vector<int> sampled_number;
           std::vector<int> population;
           for(int i = 0; i < n; ++i) {
               population.push_back(i);
           }
   
           int temp,randPos;
           for(int i = n-1; i > 0; --i) {
               randPos = (int)floor(rand_vec[n-1-i] * (i+1));
               temp = population[i];
               population[i] = population[randPos];
               population[randPos] = temp;
               sampled_number.push_back(population[i]);
               if(n-i-1 == k-1) {
               break;
               }
           }
           if(n == k) {
             sampled_number.push_back(population[0]);
           }
           E.push_back(sampled_number);
       }
       for (int i = 0; i != n; ++i) {
           IOHprofiler_uniform_rand((size_t)pow(2,k+1),(long)(k * (i+1) * 2),rand_vec);
           F.push_back(rand_vec);
       }
     }
   
     void prepare_problem() {
       set_n_k(IOHprofiler_get_number_of_variables(),k);
     }
   
     double internal_evaluate(const std::vector<int> &x) {
     
       int n = x.size();
       double result = 0;
       int index;
       for (int i = 0; i != n; ++i) {
         index= x[i];
         for (int j = 0; j != k; ++j) {
           index = index + pow(2,(j+1)) * x[E[i][j]];
         }
         result += F[i][index];
       }
   
       result = result / (double)n;
       return -result;
     }
   
     static NK_Landscapes * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new NK_Landscapes(instance_id, dimension);
     }
   };
   
   #endif
