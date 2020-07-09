
.. _program_listing_file_src_Problems_README.md:

Program Listing for File README.md
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_README.md>` (``src/Problems/README.md``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: markdown

   ## Problems of IOHexperimenter
   
   [IOHprofiler_problem](/src/Template/IOHprofiler_problem.hpp) is the base `class` of problems of __IOHexperimenter__. The property variables of problems include:
   * `problem_id`, will be assigned if the problem is added to a suite, otherwise default by 0.
   * `instance_id`,  sets transformation methods on problems. The original problem is with instance_id 1, <i>scale</i> and <i>shift</i> are applied on objectives for instance_id in [2,100], <i>XOR</i> is applied on variables for instance_id in [2,50], and <i>sigma</i> function is applied on variables for instance_id in [51,100].
   * `problem_name`
   * `problem_type`
   * `lowerbound`, is a vector of lowerbound for variables.
   * `upperbound`, is a vector of upperbound for variables.
   * `number_of_variables`, is the dimension of the problem.
   * `number_of_objectives`, is only available as 1 now. The functionality of multi-objectives is under development.
   * `best_variables`, is a vector of optimal solution, which is used to calculate the optimum. If the best_variables is not given, the optimum will be set as __DBL_MAX__.
   * `optimal`, is a vector of optimal objectives, but currently only single objective is supported.
   
   * `evaluate_int_info`, is a vector of __int__ values that are iteratively used in <i>evaluate</i>.
   * `evaluate_double_info`, is a vector of __double__ values that are iteratively used in <i>evaluate</i>.
   
   And some functions for personal experiments are supplied:
   * <i>evaluate(x)</i>, returns a vector of fitness values. The argument __x__ is a vector of variables.
   * <i>evaluate(x,y)</i>, updates __y__ with a vector of fitness values, and __x__ is a vector of variables.
   * <i>addCSVLogger(logger)</i>, assigns a __IOHprofiler_csv_logger__ class to the problem.
   * <i>clearLogger()</i>, delete logger methods of the problem.
   * <i>reset_problem()</i>, reset the history information of problem evaluations. You should call this function at first when you plan to do another test on the same problem class.
   * <i>IOHprofiler_hit_optimal()</i>, returns true if the optimum of the problem has been found.
   * <i>IOHprofiler_set_number_of_variables(number_of_variables)</i>, sets dimension of the problem.
   * <i>IOHprofiler_set_instance_id(instance_id)</i>
   
   ### Creating a problem
   __IOHexperimenter__ provides a variety of problems for testing algorithms, but it is also easy to add your own problems. Overall, to create a problem of __IOHexperimenter__, two functions need to be implemented: <i>construct functions</i> and <i>internel_evaluate</i>. Additionally, you can define <i>update_evaluate_double_info</i> and <i>update_evaluate_int_info</i> to make evluate process more efficiently.
   
   Taking the implementation of __OneMax__ as an instance, <i>construct functions</i> are as below. `problem_name` and `number_of_objectives` __must__ be set. In general, two methods of construction of the problems are given. One is constructing without giving `instance_id` and `dimension`, and the other one is with.
   ```cpp
   OneMax() {
     IOHprofiler_set_problem_name("OneMax");
     IOHprofiler_set_problem_type("pseudo_Boolean_problem");
     IOHprofiler_set_number_of_objectives(1);
     IOHprofiler_set_lowerbound(0);
     IOHprofiler_set_upperbound(1);
     IOHprofiler_set_best_variables(1);
   }
   
   OneMax(int instance_id, int dimension) {
     IOHprofiler_set_instance_id(instance_id);
     IOHprofiler_set_problem_name("OneMax");
     IOHprofiler_set_problem_type("pseudo_Boolean_problem");
     IOHprofiler_set_number_of_objectives(1);
     IOHprofiler_set_lowerbound(0);
     IOHprofiler_set_upperbound(1);
     IOHprofiler_set_best_variables(1);
     Initilize_problem(dimension);
   }
     
   ~OneMax() {};
   
   void Initilize_problem(int dimension) {
     IOHprofiler_set_number_of_variables(dimension);
     IOHprofiler_set_optimal((double)dimension);
   };
   ```
   
   The <i>internal_evaluate</i> __must__ be implemented as well. It is used during evaluate process, returning a vector of (real) objective values of the corresponding variables __x__.
   ```cpp
   std::vector<double> internal_evaluate(std::vector<int> x) {
     std::vector<double> y;
     int n = x.size();
     int result = 0;
     for (int i = 0; i != n; ++i) {
       result += x[i];
     }
     y.push_back((double)result);
     return y;
   };
   ```
   
   If you want to register your problem by `problem_name` and add it into a suite, please add functions creating instances as following codes.
   ```cpp
   static OneMax * createInstance() {
     return new OneMax();
   };
   
   static OneMax * createInstance(int instance_id, int dimension) {
     return new OneMax(instance_id, dimension);
   };
   ```
   To register the problem, you can use the <i>geniricGenerator</i> in [IOHprofiler_class_generator](/src/Template/). For example, you can use the following statement to register and create __OneMax__ ,
   ```cpp
   // Register
   static registerInFactory<IOHprofiler_problem<int>,OneMax> regOneMax("OneMax");
   // Create
   std::shared_ptr<IOHprofiler_problem<int>> problem = genericGenerator<IOHprofiler_problem<int>>::instance().create("OneMax");
   ```
