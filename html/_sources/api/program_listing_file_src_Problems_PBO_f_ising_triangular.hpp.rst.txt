
.. _program_listing_file_src_Problems_PBO_f_ising_triangular.hpp:

Program Listing for File f_ising_triangular.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_ising_triangular.hpp>` (``src/Problems/PBO/f_ising_triangular.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ISING_TRIANGULAR_H
   #define _F_ISING_TRIANGULAR_H
   
   #include "IOHprofiler_problem.h"
   
   class Ising_Triangular : public IOHprofiler_problem<int> {
   public:
   
     Ising_Triangular(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
   
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("Ising_Triangular");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~Ising_Triangular() {}
   
     int modulo_ising_triangular(int x,int N) {
       return (x % N + N) % N;
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       int i, j, neig;
       int n = x.size();
       int result = 0;
       int neighbors[3];
       int lattice_size = (int)sqrt((double)n);
       
       for (i = 0; i < lattice_size; ++i) {
         for (j = 0; j < lattice_size; ++j) {
           neighbors[0] = x[modulo_ising_triangular((i + 1), lattice_size) * lattice_size + j];
           neighbors[1] = x[i * lattice_size + modulo_ising_triangular((j + 1), lattice_size)];
           neighbors[2] = x[modulo_ising_triangular((i + 1) , lattice_size) * lattice_size + modulo_ising_triangular((j + 1), lattice_size)];
   
           for (neig=0; neig < 3; neig++) {
             result += (x[i * lattice_size + j] * neighbors[neig]) - ((1 - x[i * lattice_size + j]) * (1 - neighbors[neig]));
           }
         }
       }
       return (double)result;
     }
   
     static Ising_Triangular * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Ising_Triangular(instance_id, dimension);
     }
   };
   
   #endif
