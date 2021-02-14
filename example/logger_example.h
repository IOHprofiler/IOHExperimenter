#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

#include "ioh.hpp"
#include "ioh/suite.hpp"
#include "ioh/logger.hpp"

using logger = ioh::logger::Csv<ioh::problem::bbob::bbob_base>;

/// An example of using a csv logger class to store evaluation information during the optimization process.
void logger_example() {
  
  /// To declare a csv logger class.
  /// output_directory : "./"
  /// folder_name : "logger_example"
  /// algorithm_name : "random search'
  /// algorithm_info : "a random search for testing the bbob suite"
  auto l = std::make_shared<logger>("./", "logger_example", "random_search", "a random search for testing the bbob suite");
  
  /// To declare a bbob suite of problem {1}, intance {1, 2} and dimension {5,6}.
  ioh::suite::bbob bbob({1}, {1, 2}, {5, 6});
  
 
  std::shared_ptr<ioh::problem::bbob::bbob_base> problem;
  /// To access problem classes of the suite by using 'get_next_problem()' function.
  while ((problem = bbob.get_next_problem()) != nullptr) {
    
    l->track_problem(*problem);
    
    int budget = 100;
    int n = problem->get_number_of_variables();
    
    /// Random search on the problem with the given budget 100.
    std::vector<double> x(n);
    auto best_y = std::numeric_limits<double>::infinity();
    while(budget > 0) {
      budget--;
      for (int i = 0; i != n; i++) {
        ioh::common::Random::uniform(n, budget, x);
        x[i] = x[i] * 10 - 5;
      }
      
      /// To evalute the fitness of 'x' for the problem by using 'evaluate(x)' function.
      best_y = std::min(problem->evaluate(x), best_y);
      /// To pass current evaluation information to the logger, and to output information into csv files.
      l->do_log(problem->loggerCOCOInfo());
    }
    std::cout << "result: " << best_y << std::endl;
  }
  
  /// Close the logger after all evaluations.
  l->clear_logger();
}


