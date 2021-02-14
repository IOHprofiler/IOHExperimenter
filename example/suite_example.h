#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

#include "ioh/suite.hpp"

/// An example of using a suite class of problems.
void suite_example() {
 
  /// To declare a bbob suite of problem {1}, intance {1, 2} and dimension {5,6}.
  ioh::suite::bbob bbob({1}, {1, 2}, {5, 6});
  
  std::cout << "An example of using bbob suite" << std::endl;
  std::shared_ptr<ioh::problem::bbob::bbob_base> problem;
  
  /// To access problem classes of the suite by using 'get_next_problem()' function.
  while ((problem = bbob.get_next_problem()) != nullptr) {
    
    int runs = 5;
    while(runs-- > 0) {
      /// To output information of the current problem.
      std::cout << "Problem " << problem->get_problem_name() << ", " <<
      "Instance " << problem->get_instance_id() << "," <<
      "Dimension " << problem->get_number_of_objectives() << ", " <<
      "Run " << 5 - runs << ", ";
      
      int budget = 100;
      int n = problem->get_number_of_variables();
      
      /// Random search on the problem with the given budget 100.
      std::vector<double> x(n);
      auto best_y = std::numeric_limits<double>::infinity();
      while(budget > 0) {
        budget--;
        for (int i = 0; i != n; i++) {
          ioh::common::Random::uniform(n, budget * runs, x);
          x[i] = x[i] * 10 - 5;
        }
        
        /// To evalute the fitness of 'x' for the problem by using 'evaluate(x)' function.
        best_y = std::min(problem->evaluate(x), best_y);
      }
      
      /// To reset evaluation information as default before the next independent run.
      std::cout << "result: " << best_y << std::endl;
    }
  }
}


