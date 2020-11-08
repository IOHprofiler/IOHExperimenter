/// \file f_leading_ones_neutrality.hpp
/// \brief cpp file for class f_leading_ones_neutrality.
///
/// This file implements a LeadingOnes problem with neutrality transformation method from w-model.
/// The parameter mu is chosen as 3.
///
/// \author Furong Ye
/// \date 2019-06-27
#pragma once
#include "pbo_base.hpp"
#include "ioh/problem/utils.hpp"

namespace ioh
{
	namespace problem
	{
		namespace pbo
		{
			class LeadingOnes_Neutrality : public pbo_base
			{
			public:
				/**
				 * \brief Construct a new LeadingOnes_Neutrality object. Definition refers to https://doi.org/10.1016/j.asoc.2019.106027
				 * 
				 * \param instance_id The instance number of a problem, which controls the transformation
				 * performed on the original problem.
				 * \param dimension The dimensionality of the problem to created, 4 by default.
				 **/
				LeadingOnes_Neutrality(int instance_id = IOH_DEFAULT_INSTANCE, int dimension = IOH_DEFAULT_DIMENSION)
					: pbo_base(13, "LeadingOnes_Neutrality", instance_id)
				{
					set_best_variables(1);
					set_number_of_variables(dimension);
				}

				double internal_evaluate(const std::vector<int>& x) override
				{
					auto new_variables = utils::neutrality(x, 3);
					auto n = new_variables.size();
					auto result = 0;
					for (auto i = 0; i != n; ++i)
					{
						if (new_variables[i] == 1)
						{
							result = i + 1;
						}
						else
						{
							break;
						}
					}
					return static_cast<double>(result);
				}

				static LeadingOnes_Neutrality* create(int instance_id = IOH_DEFAULT_INSTANCE,
				                                      int dimension = IOH_DEFAULT_DIMENSION)
				{
					return new LeadingOnes_Neutrality(instance_id, dimension);
				}
			};
		}
	}
}
