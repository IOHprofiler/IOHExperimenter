
.. _program_listing_file_src_Template_IOHprofiler_random.cpp:

Program Listing for File IOHprofiler_random.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_random.cpp>` (``src/Template/IOHprofiler_random.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "IOHprofiler_random.h"
   
   long IOHprofiler_random::a = RND_MULTIPLIER; 
   long IOHprofiler_random::m = RND_MODULUS; 
   long IOHprofiler_random::q = RND_MODULUS_DIV; 
   long IOHprofiler_random::r = RND_MOD_MULTIPLIER; 
   
   int IOHprofiler_random::IOHprofiler_SHORT_LAG  = IOHPROFILER_SHORT_LAG;
   int IOHprofiler_random::IOHprofiler_LONG_LAG = IOHPROFILER_LONG_LAG;
   
   IOHprofiler_random::IOHprofiler_random(uint32_t seed) {
     for (int i = 0; i < IOHprofiler_LONG_LAG; ++i) {
       x[i] = ((double)seed) / (double)(((uint64_t)1UL << 32) - 1);
       seed = (uint32_t)1812433253UL * (seed ^ (seed >> 30)) + ((uint32_t)i + 1);
     }
     _seed_index = 0;
   }
   
   void IOHprofiler_random::IOHprofiler_random_generate() {
     for (int i = 0; i < IOHprofiler_SHORT_LAG; ++i) {
         double t = this->x[i] + this->x[i + (IOHprofiler_LONG_LAG - IOHprofiler_SHORT_LAG)];
         if (t >= 1.0)
             t -= 1.0;
         this->x[i] = t;
     }
     for (int i = IOHprofiler_SHORT_LAG; i < IOHprofiler_LONG_LAG; ++i) {
         double t = this->x[i] + this->x[i - IOHprofiler_SHORT_LAG];
         if (t >= 1.0)
             t -= 1.0;
         this->x[i] = t;
     }
     this->_seed_index = 0;
   }
   
   long IOHprofiler_random::_lcg_rand(const long &inseed) {
     
     long new_inseed =  (long)(a * (inseed - (long)floor((double)inseed / (double)q) * q) - r * (long)floor((double)inseed / (double)q));
     if (new_inseed < 0) {
       new_inseed = new_inseed + m;
     }
     return new_inseed;
   }
   
   void IOHprofiler_random::IOHprofiler_uniform_rand(const size_t &N, const long &inseed, std::vector<double> &rand_vec) {
     if (rand_vec.size() != 0) {
       std::vector<double>().swap(rand_vec);
     }
     rand_vec.reserve(N);
   
     long rand_seed[32];
     long seed;
     long rand_value;
   
     seed = inseed;
     if (seed < 0) {
       seed = -seed;
     }
     if (seed < 1) {
       seed = 1;
     }
   
     seed = inseed;
     for (int i = 39; i >= 0; --i) {
       seed = _lcg_rand(seed);
       if(i < 32) {
         rand_seed[i] = seed;
       }
     }
     
     int seed_index = 0;
     seed = rand_seed[0];
     for (int i = 0; i < N; ++i) {
       rand_value = _lcg_rand(seed);
       
       seed_index = (int)floor((double)seed / (double)67108865);
       seed = rand_seed[seed_index];
       rand_seed[seed_index] = rand_value;
   
       rand_vec.push_back((double)seed/2.147483647e9);
       if (rand_vec[i] == 0.) {
         rand_vec[i] = 1e-99;
       }
     }
   }
   
   std::vector<double> IOHprofiler_random::IOHprofiler_gauss(const size_t N, const long inseed) {
     std::vector<double> rand_vec;
     std::vector<double> uniform_rand_vec;
     rand_vec.reserve(N);
   
     long seed;
     long rand_value;
   
     seed = inseed;
     if (seed < 0) {
       seed = -seed;
     }
     if (seed < 1) {
       seed = 1;
     }
   
     IOHprofiler_uniform_rand(2 * N, seed,uniform_rand_vec);
   
     for (int i = 0; i < N; i++) {
       rand_vec.push_back(sqrt(-2 * log(uniform_rand_vec[i])) * cos(2 * IOH_PI * uniform_rand_vec[N + i]));
       if (rand_vec[i] == 0.) {
         rand_vec[i] = 1e-99;
       }
     }
     return rand_vec;
   }
   
   double IOHprofiler_random::IOHprofiler_uniform_rand() {
     if (this->_seed_index >= IOHprofiler_LONG_LAG) {
       IOHprofiler_random_generate();
     }
     return this->x[this->_seed_index++];
   }
   
   double IOHprofiler_random::IOHprofiler_normal_rand() { /* OMS: accessor - should become const */
     // double normal;
     // normal = 0.0;
     // for (int i = 0; i < 12; ++i) {
     //     normal += this->IOHprofiler_uniform_rand();
     // }
     // normal -= 6.0;
     // return normal;
     double normal;
   #ifdef IOHprofiler_NORMAL_POLAR
     const double u1 = this->IOHprofiler_uniform_rand();
     const double u2 = this->IOHprofiler_uniform_rand();
     normal = sqrt(-2 * log(u1)) * cos(2 * IOH_PI * u2);
   #else
     int i;
     normal = 0.0;
     for (i = 0; i < 12; ++i) {
         normal += this->IOHprofiler_uniform_rand();
     }
     normal -= 6.0;
   #endif
     return normal;
   }
