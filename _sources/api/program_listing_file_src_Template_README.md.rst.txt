
.. _program_listing_file_src_Template_README.md:

Program Listing for File README.md
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_README.md>` (``src/Template/README.md``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: markdown

   ## Source codes of IOHexperimenter
   
   [This folder] includes common used codes of __IOHexperimenter__. For details of `IOHprofiler_problem` and `IOHprofiler_suite`, please visit [problem page](/src/Problems/) and [suite page](/src/Suites).
   
   ### IOHprofiler_transformation
   
   IOHprofiler_transformation` defines methods of transformation applied on `IOHprofiler_problem`. For the definition of methods, please visit [wikipage](https://iohprofiler.github.io/Benchmark/Transformation/)
   
   ### IOHprofiler_random
   
   __IOHexperimenter__ supplies methods of generating random numbers, which can be used to reproduce same experiments with the same random seed.
   
   `IOHprofiler_random` is a `class` of random methods, you can construct it with a specified random seed as:
   ```cpp
   IOHprofiler_random(long seed);
   ```
   
   And two random number generator methods are as follow.
   * <i>double IOHprofiler_uniform_rand()</i>, returns a random number from standard uniform distribution.
   * <i>double IOHprofiler_normal_rand()</i>, returns a random number from standard normal distribution.
   
   ### IOHprofiler_observer
   
   `IOHprofiler_observer` defines triggers of recording evaluations, and __logger__ classes inherit it to set up the time of recording.
   
   Four strategies of recording evaluations are available,
   * __complete tracking__, provides the highest granularity, by storing information for each function evaluation. Use <i>set_complete_flag(true)</i> to enable this strategy,
   * __interval tracking__, stores information for each $\tau$-th function evaluation. Use <i>set_interval()</i> to set $\tau > 0$ to enable this stragety.
   * __target-based tracking__, stores information for each iteration in which the best-so-far fitness improved. Use <i>set_update_flag(true)</i> to enbale this stragety.
   * __time-based tracking__, stores information when the user-specified running time budgets are reached. These budget are evenly spaced in log-10 scale.
   
   ### IOHprofier_class_generator
   
   `IOHprofiler_class_generator` supplied methods of registering and creating `IOHprofiler_problem` and `IOHprofiler_suite`.
