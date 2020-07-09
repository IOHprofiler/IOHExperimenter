
.. _program_listing_file_src_Template_Experiments_README.md:

Program Listing for File README.md
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_Experiments_README.md>` (``src/Template/Experiments/README.md``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: markdown

   ## Experiment and configuration of IOHexperimenter.
   
   __IOHexperimenters__ provides interface of using configuration files. After setting the path of the configuration file and giving an algorithm, the `IOHprofiler_experimenter` can organize experiments automatically.
   
   `IOHprofiler_configuration` reads configuration information from a given file, and these information can be used in `IOHprofiler_experimenter` to generate __IOHprofiler_suite__ and __IOHprofiler_csv_logger__.
   
   For the content of configuration file, it consists of three sections:
   
   __suite__ configures the problems to be tested.
   * __suite_name__, is the name of the suite to be used. Please make sure that the suite with the corresponding name is registered.
   * __problem_id__, configures problems to be tested. Note that id of problems are configured by the suite, please make sure that id is within the valid range.
   * __instance_id__, configures the transformation methods applied on problems. 
       For `PBO`:
       * `1` : no transformer operations on the problem.
       * `2-50` :  XOR and SHIFT operations are applied on the problem.
       * `51-100`: SIGMA and SHIFT operations are applied on the problem.
   * __dimension__, configures dimension of problems to be tested. Note that allowed dimension is not larger than 20000.
   
   __logger__ configures the setting of output csv files.
   * __result_foler__ is the directory of the folder where sotres output files.
   * __algorithm_name__, is the name of the algorithm, which is used when generating '.info' files.
   * __algorithm_info__, is additional information of the algorithm, which is used when generating '.info' files.
   
   __observer__ configures parameters of `IOHprofiler_server`, which is used in `IOHprofiler_csv_logger`,
   * __complete_triggers__ is the switch of `.cdat` files, which works with __complete tracking__ strategy. Set it as `TRUE` or `true` if you want to output `.cdat` files.
   * __update_triggers__ is the switch of `.dat` files, which works with __target-based strategy__ strategy. Set it as `TRUE` or `true` if you want to output <i>.dat</i>` files.
   * __number_interval_triggers__ configures the `.idat` files, which works with __interval tracking__  number_target_triggers sets the value of the frequecny. If you do not want to generate `.idat` files, set `number_target_triggers` as 0.
   * __number_target_triggers__ configures the `.tdat` files, which works with __time-based tracking__ strategy.
   * __base_evaluation_triggers__ configures the `.tdat` files, which works with __time-based tracking__ strategy. To switch off `.tdat` files, set both __number_target_triggers__ and __base_evaluation_triggers__ as 0. For the detail of __time-based tracking__, please with the [wiki page](https://iohprofiler.github.io/IOHexperimenter/Loggers/Observer/).
   
   And for the function of algorithms, a shared smart pointer of `IOHprofiler_problem` should be one and the only one argument of it. An example of declaration of an evolutionary algorithm is as follow,
   ```cpp
   void evolutionary_algorithm(std::shared_ptr<IOHprofiler_problem<int>> problem)
   ```
   
   In addition, `IOHprofiler_experimenter` allows to test each problem mutliple times by using the following statements.
   ```cpp
   experimenter._set_independent_runs(1);
   ```
