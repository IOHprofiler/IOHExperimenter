
.. _program_listing_file_src_Problems_PBO_f_ising_ring.hpp:

Program Listing for File f_ising_ring.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_PBO_f_ising_ring.hpp>` (``src/Problems/PBO/f_ising_ring.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_ISING_RING_H
   #define _F_ISING_RING_H
   
   #include "IOHprofiler_problem.h"
   
   class Ising_Ring : public IOHprofiler_problem<int> {
   public:
     Ising_Ring(int instance_id, int dimension) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_name("Ising_Ring");
       IOHprofiler_set_problem_type("pseudo_Boolean_problem");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(0);
       IOHprofiler_set_upperbound(1);
       IOHprofiler_set_best_variables(1);
       IOHprofiler_set_number_of_variables(dimension);
     }
   
     ~Ising_Ring() {}
   
     int modulo_ising_ring(int x,int N) {
       return (x % N + N) %N;
     }
   
     double internal_evaluate(const std::vector<int> &x) {
       int result = 0, n = x.size();
   
       int neig;
       for (int i = 0; i < n; ++i) {
         neig = x[modulo_ising_ring((i - 1) , n)];
         result += (x[i] * neig) - ((1 - x[i]) * (1 - neig));
       }
       return (double)result;
     }
   
     static Ising_Ring * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Ising_Ring(instance_id, dimension);
     }
   };
   
   #endif
