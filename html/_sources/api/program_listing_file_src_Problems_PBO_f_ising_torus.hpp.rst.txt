
.. _program_listing_file_src_Problems_PBO_f_ising_torus.hpp:

Program Listing for File f_ising_torus.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_ising_torus.hpp>` (``src/Problems/PBO/f_ising_torus.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ISING_TWOD_H
   #define _F_ISING_TWOD_H
   
   #include "IOHprofiler_problem.h"
   
   class Ising_Torus : public IOHprofiler_problem<int> {
   public:
     Ising_Torus(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("Ising_Torus");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~Ising_Torus() {}
   
     int modulo_ising_torus(int x,int N) {
       return (x % N + N) %N;
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       int n = x.size();
       int i, j, neig;
       int result= 0;
       int neighbors[2];
       int lattice_size = (int)sqrt((double)n);
       
       if(floor(sqrt((double)n))!=sqrt((double)n)) {
         IOH_error("Number of parameters in the Ising square problem must be a square number");
       }
   
       for(i = 0; i < lattice_size; ++i) {
         for (j = 0; j < lattice_size; ++j) {
           neighbors[0]  = x[modulo_ising_torus(i + 1, lattice_size) * lattice_size + j];
           neighbors[1]  = x[i * lattice_size + modulo_ising_torus((j + 1) , lattice_size)];
           for (neig = 0; neig < 2; neig++) {
             result += (x[i*lattice_size + j] * neighbors[neig]) - ((1 - x[i * lattice_size + j]) * (1 - neighbors[neig]));
           }
         }
       }
   
       return (double)result;
     }
   
     static Ising_Torus * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION)  {
       return new Ising_Torus(instance_id, dimension);
     }
   };
   
   #endif
