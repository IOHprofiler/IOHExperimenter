
.. _program_listing_file_src_Problems_PBO_f_MIS.hpp:

Program Listing for File f_MIS.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_MIS.hpp>` (``src/Problems/PBO/f_MIS.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_MIS_H
   #define _F_MIS_H
   
   #include "IOHprofiler_problem.h"
   
   class MIS : public IOHprofiler_problem<int> {
   public:
     MIS(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("MIS");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~MIS() {}
     
     int isEdge (int i, int j, size_t problem_size) {
       if (i!=problem_size/2 && j==i+1) {
         return 1;
       } else if (i<=((int)problem_size/2)-1 && j==i+((int)problem_size/2)+1 ) {
         return 1;
       } else if (i<=((int)problem_size/2) && i>=2 && j==i+((int)problem_size/2)-1) {
         return 1;
       } else {
         return 0;
       }
     }
   
   
   double internal_evaluate(const std::vector<int> &x) {
       int i, j,index;
       int n = x.size();
       int result= 0;
       int num_of_ones=0;
       int sum_edges_in_the_set=0;
       int number_of_variables_even=(int)n;
       std::vector<int> ones_array(number_of_variables_even+1);
     
       if (n%2!=0) {
           number_of_variables_even=(int)n-1;
       }
      
       for (index=0; index<number_of_variables_even; index++) {
         if (x[index]==1) {
           ones_array[num_of_ones] = index;
           num_of_ones+=1;
         }
       }
   
       for (i=0; i<num_of_ones; i++) {
         for (j=i+1; j<num_of_ones; j++) {
           if (isEdge(ones_array[i]+1,ones_array[j]+1,(size_t)number_of_variables_even)==1) {
             sum_edges_in_the_set+=1;
           }
         }
       }
       result=num_of_ones - (number_of_variables_even*sum_edges_in_the_set);
       return (double)result;
     }
   
     static MIS * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new MIS(instance_id, dimension);
     }
   };
   
   #endif
