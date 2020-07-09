
.. _program_listing_file_src_Template_IOHprofiler_random.h:

Program Listing for File IOHprofiler_random.h
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_random.h>` (``src/Template/IOHprofiler_random.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _IOHPROFILER_RANDOM_H
   #define _IOHPROFILER_RANDOM_H
   
   #include "IOHprofiler_common.h"
   #define IOHprofiler_NORMAL_POLAR /* Use polar transformation method */
   
   #define IOH_PI 3.14159265358979323846
   #define DEFAULT_SEED 1000
   
   
   #define RND_MULTIPLIER 16807
   #define RND_MODULUS 2147483647
   #define RND_MODULUS_DIV 127773
   #define RND_MOD_MULTIPLIER 2836
   
   #define IOHPROFILER_SHORT_LAG  273
   #define IOHPROFILER_LONG_LAG 607
   
   class IOHprofiler_random {
   public:
     IOHprofiler_random(uint32_t seed = DEFAULT_SEED);
   
     void IOHprofiler_random_generate();
   
     static long _lcg_rand(const long &inseed);
   
     static void IOHprofiler_uniform_rand(const size_t &N, const long &inseed, std::vector<double> &rand_vec);
     
     static std::vector<double> IOHprofiler_gauss(const size_t N, const long inseed);
     
     double IOHprofiler_uniform_rand();
     
     double IOHprofiler_normal_rand();
   
   private:
     size_t _seed_index;
   
     static long a; 
     static long m; 
     static long q; 
     static long r; 
   
     static int IOHprofiler_SHORT_LAG;
     static int IOHprofiler_LONG_LAG;
     double x[607];
   };
   
   #endif // _IOHPROFILER_RANDOM_H
