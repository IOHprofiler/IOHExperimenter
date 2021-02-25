#pragma once

#include "ioh/problem/bbob/bbob_base.hpp"

namespace ioh::problem::bbob
{

    class GriewankRosenBrock final: public BBOB, AutomaticFactoryRegistration<GriewankRosenBrock, RealProblem>
    
        {
         protected:
             std::vector<double> evaluate(std::vector<double>& x) override {}
         public:        
             GriewankRosenBrock(const int instance, const int n_variables) :                 
                BBOB(19, instance, n_variables, "GriewankRosenBrock") {}
        
        };
}