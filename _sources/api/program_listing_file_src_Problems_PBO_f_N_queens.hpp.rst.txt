
.. _program_listing_file_src_Problems_PBO_f_N_queens.hpp:

Program Listing for File f_N_queens.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_N_queens.hpp>` (``src/Problems/PBO/f_N_queens.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_NQUEENS_H
   #define _F_NQUEENS_H
   
   #include "IOHprofiler_problem.h"
   
   class NQueens : public IOHprofiler_problem<int> {
   public:
     NQueens(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("NQueens");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~NQueens() {}
   
     void customize_optimal() {
       IOHprofiler_set_optimal(sqrt( (double) IOHprofiler_get_number_of_variables() ) );
     }
   
     double _max(double element1 , double element2) {
       if (element1>element2) {
         return element1;
       } else {
         return element2;
       }
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       
       double result;
       int n = x.size();
       int index,j,i,k,l;
       int N_queens=(int)(sqrt((double)n)+0.5);
       int number_of_queens_on_board = 0;
       double k_penalty=0.0;
       double l_penalty =0.0;
       double raws_penalty=0.0;
       double columns_penalty=0.0;
       int indx=0;
       float C = (float)N_queens;
   
       if (floor(sqrt((double)n))!=sqrt((double)n)) {
         IOH_error("Number of parameters in the N Queen problem must be a square number");
       }
   
       for (index=0; index<n; index++) {
           if (x[index]==1){
             number_of_queens_on_board+=1;
           }
       }
   
       for (j=1; j <=N_queens; j++) {
           double sum_column = 0.0;
           for (i=1; i<=N_queens; i++) {
               indx=((i-1)*N_queens) + ((j-1)%N_queens);
               sum_column+=(double)x[indx];
           }
           columns_penalty+=_max(0.0, (-1.0+sum_column));
       }
   
       for (i=1; i<=N_queens; i++) {
           double sum_raw = 0.0;
           /*double sum_k = 0.0;
           double sum_l = 0.0;*/
           for (j=1; j <=N_queens; j++) {
               indx=((i-1)*N_queens) + ((j-1)%N_queens);
               sum_raw+=(double)x[indx];
           }
           raws_penalty+=_max(0.0, (-1.0+sum_raw));
       }
   
       for (k=2-N_queens; k<=N_queens-2; k++) {
           double sum_k=0.0;
           for (i=1; i<=N_queens; i++) {
               if (k+i>=1 && k+i<=N_queens) {
                   indx=((i-1)*N_queens) + ((k+i-1)%N_queens);
                   sum_k += (double)x[indx];
               }
           }
           k_penalty+=_max(0.0, (-1.0+sum_k));
       }
       for (l=3; l<=2*N_queens-1; l++)  {
           double sum_l=0.0;
           for (i=1; i<=N_queens; i++) {
               if (l-i>=1 && l-i<=N_queens) {
                   indx=((i-1)*N_queens) + ((l-i-1)%N_queens);
                   sum_l += (double)x[indx];
               }
           }
           l_penalty+=_max(0.0, (-1.0+sum_l));
       }
       result = (double) (number_of_queens_on_board - (C*raws_penalty) - (C*columns_penalty) -(C*k_penalty) - (C*l_penalty));
       return (double)result;
     }
   
     static NQueens * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new NQueens(instance_id, dimension);
     }
   };
   
   #endif
