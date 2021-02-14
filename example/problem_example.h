#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

#include "ioh.hpp"
#include "ioh/logger.hpp"
#include "ioh/problem/pbo.hpp"

/// An example of using problem classes, and a csv logger is used to store evaluation information.
void problem_example() {
  
  /// To declare a OneMax class;
  ioh::problem::pbo::OneMax om;
  std::vector<int> instance{1,2,3,51,52};
  std::vector<int> dimension{50,100};
  
  /// To declare a csv logger class.
  /// output_directory : "./"
  /// folder_name : "problem_example"
  /// algorithm_name : "random search'
  /// algorithm_info : "a random search for testing OneMax"
  auto l = std::make_shared<logger>("./", "problem_example", "random_search", "a random search for testing OneMax");
  
  std::cout << "An example of testing onemax" << std::endl;
  
  /// To access problem classes of the suite by using 'get_next_problem()' function.
  for(std::vector<int>::iterator d = dimension.begin(); d != dimension.end(); ++d) {
    for(std::vector<int>::iterator i = instance.begin(); i != instance.end(); ++i) {
      
      om.set_instance_id(*i);
      om.set_number_of_variables(*d);
      
      int runs = 5;
      while(runs-- > 0 ) {
        l->track_problem(om.get_problem_id(),om.get_number_of_variables(), om.get_instance_id(),om.get_problem_name(),om.get_optimization_type());
        /// To output information of the current problem.
        std::cout << "OneMax " << om.get_problem_name() << ", " <<
        "Instance " << om.get_instance_id() << "," <<
        "Dimension " << om.get_number_of_variables() << ", " <<
        "Run " << 5 - runs << ", ";
        
        int budget = 100;
        int n = om.get_number_of_variables();
        
        /// Random search on the problem with the given budget 100.
        std::vector<int> x(n);
        std::vector<double> r(n);
        auto best_y = -std::numeric_limits<double>::infinity();
        while(budget > 0) {
          budget--;
          ioh::common::Random::uniform(n, budget * runs, r);
          for (int i = 0; i != n; i++) {
            x[i] = static_cast<int>(r[i] * 2);
          }
          
          /// To evalute the fitness of 'x' for the problem by using 'evaluate(x)' function.
          best_y = std::max(om.evaluate(x), best_y);
          /// To pass current evaluation information to the logger, and to output information into csv files.
          l->do_log(om.loggerInfo());
        }
        std::cout << "result: " << best_y << std::endl;
        
        /// To reset evaluation information as default before the next independent run.
        om.reset_problem();
      }
    }
  }
  /// Close the logger after all evaluations.
  l->clear_logger();
}


