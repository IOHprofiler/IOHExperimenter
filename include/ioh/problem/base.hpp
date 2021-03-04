#pragma once

#include <limits>
#include <ostream>
#include <string>
#include <vector>


#include "transformation.hpp"
#include "ioh/common.hpp"

namespace ioh {
    namespace problem {


        class Problem {
             /*
             * What blocks has a problem?:
             *  - MetaData:
             *      - suite_related: id, suite_name -> we need an Wrapper for logger, to make a `Dummy Suite` (for info file)
             *      - problem_related: instance, dim~number_of_variables, name, minmax, datatype, objectives -> call constructor)
             *          class Solution(a good name) { -> get target?
             *              x, y
             *          }
             *         (think of paralization in python)
             *  - Constraints      
             *  - State:`                                      (protected/private)              (public) 
             *      - evaluations, optimalFound(method), current_best_internal (Solution), current_best (Solution) (this is transformed)
             *      - Logger info:                                                      (this is return by evaluate)
             *           R"#("function evaluation" "current f(x)" "best-so-far f(x)" "current af(x)+b" "best af(x)+b")#";
             *                                      values that are the same across instances (iid independent: untransformed); transformed(iid dependent)
             *                                           for BBOB these are also scaled with optimum (goes to zero, not for pbo)
             *          Make a different implementation for current_best & current_best_transformed for COCO and PBO (default)
             *  -
             *
             *  void calc_optimal()
             *  reset
                virtual void prepare() {}
                
                virtual void customize_optimal() -> child class
                virtual void transform_variables() -> self
                virtual void transform_objective() 
                double evaluate -> operator overload ()
                protected virtual double evaluate(const std::vector<InputType>& x) = 0;

                accessor naming:
                    named after the private variable
                members:
                    not: problem_name but: name
                    
             */

        };

     








        /** \brief A base class for IOH problems.
         * Basic structure for IOHexperimenter, which is used for generating benchmark problems.
         * To define a new problem, the `internal_evaluate` method must be defined, 
         * where the definition of the problem should locate.  The problem sets as maximization
         * by default. If the 'best_variables' are given, the optimal of the problem will be 
         * calculated with the 'best_variables'; or you can set the optimal by defining the 
         * 'customized_optimal' function; otherwise, the optimal is set as min()(max()) for 
         * maximization(minimization). If additional calculation is needed by `internal_evaluate`,
         * you can configure it in `prepare_problem()`.
         */
        template <class InputType>
        class base {
            int problem_id; /// < problem id
            int instance_id;
            /// < evaluate function is validated with get and dimension. set default to avoid invalid class.

            std::string problem_name;
            std::string problem_type; /// todo. make it as enum. -> suite_name

            common::OptimizationType maximization_minimization_flag; 

            int number_of_variables;
            /// < evaluate function is validated with get and dimension. set default to avoid invalid class.
            int number_of_objectives;

            std::vector<InputType> lowerbound;
            std::vector<InputType> upperbound;

            std::vector<InputType> best_variables; /// todo. comments, rename?
            std::vector<InputType> best_transformed_variables; 
            std::vector<double> optimal;
            /// todo. How to evluate distance to optima. In global optima case, which optimum to be recorded.
            bool optimalFound;

            std::vector<double> raw_objectives;
            /// < to record objectives before transformation.
            std::vector<double> transformed_objectives;
            /// < to record objectives after transformation.
            int transformed_number_of_variables;
            /// < intermediate variables in evaluate.
            std::vector<InputType> transformed_variables;
            /// < intermediate variables in evaluate.

            /// todo. constrainted optimization.
            /// std::size_t number_of_constraints;

            int evaluations; /// < to record optimization process.
            std::vector<double> best_so_far_raw_objectives;
            /// < to record optimization process.
            int best_so_far_raw_evaluations;
            /// < to record optimization process.
            std::vector<double> best_so_far_transformed_objectives;
            /// < to record optimization process.
            int best_so_far_transformed_evaluations;
            /// < to record optimization process.

        public:
            double t{};

            base(int instance_id = IOH_DEFAULT_INSTANCE,
                 int dimension = IOH_DEFAULT_DIMENSION)
                : problem_id(IOH_DEFAULT_PROBLEM_ID),
                  instance_id(instance_id),
                  maximization_minimization_flag(
                      common::OptimizationType::maximization),
                  number_of_variables(dimension),
                  number_of_objectives(1),
                  lowerbound(std::vector<InputType>(number_of_variables)),
                  upperbound(std::vector<InputType>(number_of_variables)),
                  optimalFound(false),
                  raw_objectives(std::vector<double>(number_of_objectives)),
                  transformed_objectives(
                      std::vector<double>(number_of_objectives)),
                  transformed_number_of_variables(0),
                  evaluations(0),
                  best_so_far_raw_objectives(
                      std::vector<double>(number_of_objectives)),
                  best_so_far_raw_evaluations(0),
                  best_so_far_transformed_objectives(
                      std::vector<double>(number_of_objectives)),
                  best_so_far_transformed_evaluations(0) {
            }

            base(const base &) = delete;
            base &operator=(const base &) = delete;

            /** \todo to support multi-objective optimization
             * \fn virtual std::vector<double> internal_evaluate_multi(std::vector<InputType> x)
             * \brief A virtual internal evaluate function.
             *
             * The internal_evaluate function is to be used in evaluate function.
             * This function must be decalred in derived function of new problems.
             */
            // virtual std::vector<double> internal_evaluate_multi (const std::vector<InputType> &x) {
            //   std::vector<double> result;
            //   std::cout << "No multi evaluate function defined" << std::endl;
            //   return result;
            // };

            /**
             * \fn double internal_evaluate(std::vector<InputType> x)
             * \brief A virtual internal evaluate function.
             * 
             * The internal_evaluate function is to be used in evaluate function.
             * This function must be declared in derived function of new problems.
             */
            virtual double internal_evaluate(const std::vector<InputType>& x) = 0;

            /**
             * \fn virtual void prepare_problem()
             * \brief A virtual function for additional preparation of the problem.
             * 
             * Additional preparation, such as calculatng values of parameters based on
             * problem_id, dimension, instance id, etc., can be done in this function.
             */
            virtual void prepare_problem(){}

            /** \todo to support multi-objective optimization
             * \fn std::vector<double> evevaluate_multialuate(std::vector<InputType> x)
             * \brife A common function for evaluating fitness of problems.
             * 
             * Raw evaluate process, tranformation operations, and logging process are excuted 
             * in this function.
             * \param x A InputType vector of variables.
             * \return A double vector of objectives.
             */
            // std::vector<double> evaluate_multi(std::vector<InputType> x) {
            //   ++this->evaluations;

            //   transformation.variables_transformation(x,this->problem_id,this->instance_id,this->problem_type);
            //   this->raw_objectives = internal_evaluate_multi(x);

            //   this->transformed_objectives = this->raw_objectives;
            //   transformation.objectives_transformation(x,this->transformed_objectives,this->problem_id,this->instance_id,this->problem_type);
            //   if (compareObjectives(this->transformed_objectives,this->best_so_far_transformed_objectives,this->maximization_minimization_flag)) {
            //     this->best_so_far_transformed_objectives = this->transformed_objectives;
            //     this->best_so_far_transformed_evaluations = this->evaluations;
            //     this->best_so_far_raw_objectives = this->raw_objectives;
            //     this->best_so_far_raw_evaluations = this->evaluations;

            //   }

            //   if (compareVector(this->transformed_objectives,this->optimal)) {
            //     this->optimalFound = true;
            //   }

            //   return this->transformed_objectives;
            // }

            /** \fn double evaluate(std::vector<InputType> x)
             *  \brief A common function for evaluating fitness of problems.
             * 
             * Raw evaluate process, tranformation operations, and logging process are 
             * executed in this function.
             * \param x A InputType vector of variables.
             * \return A double vector of objectives.
             */
            double evaluate(std::vector<InputType> x) {
                assert(!this->raw_objectives.empty());
                assert(this->transformed_objectives.size() == this->raw_objectives.size());

                t = static_cast<double>(this->evaluations);

                ++this->evaluations;

                if (static_cast<int>(x.size()) != this->number_of_variables) {
                    common::log::warning("The dimension of solution is incorrect.");
                    // common::log::warning(std::to_string(x.size()));
                    // common::log::warning(std::to_string(this->number_of_variables));

                    if (this->maximization_minimization_flag == common::OptimizationType::maximization) {
                        this->raw_objectives[0] = std::numeric_limits<double>::lowest();
                        this->transformed_objectives[0] = std::numeric_limits<double>::lowest();
                    } else {
                        this->raw_objectives[0] = std::numeric_limits<double>::max();
                        this->transformed_objectives[0] = std::numeric_limits<double>::max();
                    }
                    return this->transformed_objectives[0];
                }

                variables_transformation(x, problem_id, instance_id);
                                             

                this->raw_objectives[0] = this->internal_evaluate(x);
                this->transformed_objectives[0] = this->raw_objectives[0];

                objectives_transformation(x, transformed_objectives, problem_id,
                                          instance_id);

                if (common::compare_objectives(this->transformed_objectives, this->best_so_far_transformed_objectives,
                                               this->maximization_minimization_flag)
                ) {
                    this->best_so_far_transformed_objectives = this->transformed_objectives;
                    this->best_so_far_transformed_evaluations = this->evaluations;
                    this->best_so_far_raw_objectives = this->raw_objectives;
                    this->best_so_far_raw_evaluations = this->evaluations;
                }

                if (common::compare_vector(this->transformed_objectives,
                                           this->optimal)) {
                    this->optimalFound = true;
                }
                return this->transformed_objectives[0];
            }

            /**
             * \fn void objectives_transformation(const std::vector<InputType>& x, 
             *                                     std::vector<double>& y,
             *                                     const int transformation_id, 
             *									   const int instance_id)
             * \brief A virtual transformation function on objective values.
             * \param x input variables
             * \param y the objective values of the input x
             * \param transformation_id transformation id
             * \param instance_id instance id
             * The objectives_transformation function is to be used after 
             * `internal_evaluate` function. The objective values resulting from 
             * `internal_evaluate` will be transformed by this function.
             */
            virtual void objectives_transformation(
                const std::vector<InputType> &x,
                std::vector<double> &y,
                const int transformation_id,
                const int instance_id) {
            }

            /**
             * \fn void variables_transformation(std::vector<InputType>& x, 
             *                                    const int transformation_id,
             *                                    const int instance_id)
             * \brief A virtual transformation function on input variables.
             * \param x input variables
             * \param transformation_id transformation id
             * \param instance_id instance id
             * 
             * The variables_transformation function is to be used before 
             * `internal_evaluate`. The transformed input variables be evaluated by
             * `internal_evaluate` function.
             */
            virtual void variables_transformation(std::vector<InputType> &x,
                                                  const int transformation_id,
                                                  const int instance_id) {
            }

            /** \fn virtual void customized_optimal()
             *
             * A virtual function to customize optimal of the problem.
             */
            virtual void customize_optimal() {
            }

            /** \fn void calc_optimal()
             *
             * A function to calculate the optimum of the problem.
             * It will be invoked after setting `number_of_variables` or `instance_id`.
             */
            void calc_optimal() {
                if (static_cast<int>(this->best_variables.size()) == this->number_of_variables) {
                    /// todo. Make Exception.
                    /// Do not apply transformation on best_variables as calculating the optimum
                    if (this->number_of_objectives == 1)

                        this->optimal[0] = internal_evaluate(this->best_variables);
                    else
                        common::log::error(
                            "Multi-objectives optimization is not supported now.");

                    objectives_transformation(this->best_variables,
                                              this->optimal,
                                              this->problem_id,
                                              this->instance_id);
                } else {
                    this->optimal.clear();
                    for (auto i = 0; i < this->number_of_objectives; ++i
                    ) {
                        if (this->maximization_minimization_flag ==
                            common::OptimizationType::maximization)
                            this->optimal.push_back(
                                std::numeric_limits<double>::max());
                        else
                            this->optimal.push_back(
                                std::numeric_limits<double>::lowest());
                    }
                    customize_optimal();
                }
            }

            /** \todo  To support constrained optimization.
             */
            // virtual std::vector<double> constraints() {
            //   std::vector<double> con;
            //   printf("No constraints function defined\n");
            //   return con;
            // };

            /** \fn void reset_problem()
             *
             * \brief Reset problem as the default condition before doing evaluating.
             */
            void reset_problem() {
                this->evaluations = 0;
                this->best_so_far_raw_evaluations = 0;
                this->best_so_far_transformed_evaluations = 0;
                this->optimalFound = false;
                for (auto i = 0; i != this->number_of_objectives; ++i) {
                    if (this->maximization_minimization_flag == common::OptimizationType::maximization) {
                        this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::lowest();
                        this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::lowest();
                    } else {
                        this->best_so_far_raw_objectives[i] = std::numeric_limits<double>::max();
                        this->best_so_far_transformed_objectives[i] = std::numeric_limits<double>::max();
                    }
                }
                this->prepare_problem();
                this->calc_optimal(); // you already know this
            }

            /** \fn std::vector<std::variant<int,double,std::string>> loggerInfo()
             *
             * Return a vector logger_info may be used by loggers.
             * logger_info[0] evaluations
             * logger_info[1] precision
             * logger_info[2] best_so_far_precision
             * logger_info[3] transformed_objective
             * logger_info[4] best_so_far_transformed_objectives
             */
            std::vector<double> loggerCOCOInfo() const {
                std::vector<double> logger_info(5);
                logger_info[0] = static_cast<double>(this->evaluations);
                logger_info[1] = this->transformed_objectives[0] - this->optimal[0]; // fopt
                logger_info[2] = this->best_so_far_transformed_objectives[0] - this->optimal[0];
                logger_info[3] = this->transformed_objectives[0];
                logger_info[4] = this->best_so_far_transformed_objectives[0];
                return logger_info;
            }

            /** \fn std::vector<std::variant<int,double,std::string>> loggerInfo()
             *
             * Return a vector logger_info may be used by loggers.
             * logger_info[0] evaluations
             * logger_info[1] raw_objectives
             * logger_info[2] best_so_far_raw_objectives
             * logger_info[3] transformed_objective
             * logger_info[4] best_so_far_transformed_objectives
             */
            std::vector<double> loggerInfo() const {
                std::vector<double> logger_info(5);
                logger_info[0] = static_cast<double>(this->evaluations);
                logger_info[1] = this->raw_objectives[0];
                logger_info[2] = this->best_so_far_raw_objectives[0];
                logger_info[3] = this->transformed_objectives[0];
                logger_info[4] = this->best_so_far_transformed_objectives[0];
                return logger_info;
            }

            /** \fn hit_optimal()
             *
             * \brief Detect if the optimum has been found.
             */
            bool hit_optimal() const {
                return this->optimalFound;
            };

            /** \fn int get_problem_id()
             * \brief Return problem id.
             */
            int get_problem_id() const {
                return this->problem_id;
            }

            /** \fn void set_problem_id()
             * \brief set problem id
             * 
             * \param problem_id problem id
             */
            void set_problem_id(int problem_id) {
                this->problem_id = problem_id;
            }

            /** \fn int get_instance_id()
             * \brief Return instance id.
             */
            int get_instance_id() const {
                return this->instance_id;
            }

            /** \fn set_instance_id(int instance_id)
             *
             * Set `instance_id` of the problem. 
             * Because `optimal` will be updated as `instanced_id` being updated.
             * `calc_optimal()` is revoked here.
             * 
             * \param instance_id 
             */
            void set_instance_id(int instance_id) {
                this->instance_id = instance_id;
                this->prepare_problem();
                this->calc_optimal();
            }

            /** \fn std::string get_problem_name()
             * \brief Return problem name.
             */
            std::string get_problem_name() const {
                return this->problem_name;
            }

            /** \fn void set_problem_name(std::string problem_name)
             * \brief Set problem name
             *
             * \param problem_name problem name
             */
            void set_problem_name(std::string problem_name) {
                this->problem_name = problem_name;
            }

            /** \fn std::string get_problem_type()
             * \brief Return problem type.
             */
            std::string get_problem_type() const {
                return this->problem_type;
            }

            /** \fn void set_problem_type(std::string problem_type)
             * \brief Set problem type
             *
             * \param problem_type problem type
             */
            void set_problem_type(std::string problem_type) {
                this->problem_type = problem_type;
            }

            /** \fn std::vector<InputType> get_lowerbound()
             * \brief Return lowerbound of input variables.
             */
            std::vector<InputType> get_lowerbound() const {
                return this->lowerbound;
            }

            /** \fn void set_lowerbound(const std::vector<InputType>& lowerbound)
             * \brief Set the lowerbound of input variables.
             * 
             * With this function, lowerbound of input variables at every index will be identical with the given value.
             * \param lowerbound lowerbound
             */
            void set_lowerbound(InputType lowerbound) {
                std::vector<InputType>().swap(this->lowerbound);
                this->lowerbound.reserve(this->number_of_variables);
                for (auto i = 0; i < this->number_of_variables; ++i) {
                    this->lowerbound.push_back(lowerbound);
                }
            }

            /** \fn void set_lowerbound(const std::vector<InputType>& lowerbound)
             * \brief Set the lowerbound of input variables.
             * 
             * \param lowerbound lowerbound
             */
            void set_lowerbound(const std::vector<InputType> &lowerbound) {
                this->lowerbound = lowerbound;
            }

            /** \fn std::vector<InputType> std::vector<InputType> get_upperbound()
             * \brief Return lowerbound of input variables.
             */
            std::vector<InputType> get_upperbound() const {
                return this->upperbound;
            }

            /** \fn void set_upperbound(const std::vector<InputType>& upperbound)
             * \brief Set the upperbound of input variables.
             * 
             * With this function, upperbound of input variables at every index will be identical with the given value.
             * \param upperbound upperbound
             */
            void set_upperbound(InputType upperbound) {
                std::vector<InputType>().swap(this->upperbound);
                this->upperbound.reserve(this->number_of_variables);
                for (auto i = 0; i < this->number_of_variables; ++i) {
                    this->upperbound.push_back(upperbound);
                }
            }

            /** \fn void set_upperbound(const std::vector<InputType>& upperbound)
             * \brief Set the upperbound of input variables.
             * 
             * \param upperbound upperbound
             */
            void set_upperbound(const std::vector<InputType> &upperbound) {
                this->upperbound = upperbound;
            }

            /** \fn int get_number_of_variables()
             * \brief Return dimension of the problem.
             */
            int get_number_of_variables() const {
                return this->number_of_variables;
            }

            /** \fn set_number_of_variables(int number_of_variables)
             * 
             * To set number_of_variables of the problem. When the number_of_variables
             * is updated, `bet_variables`, `lowerbound`, `upperbound`, and `optimal` 
             * need to be updated as well.
             *
             * \param number_of_variables
             */
            void set_number_of_variables(int number_of_variables) {
                this->number_of_variables = number_of_variables;
                if (this->best_variables.size() != 0) {
                    this->set_best_variables(this->best_variables[0]);
                }
                if (this->lowerbound.size() != 0) {
                    this->set_lowerbound(this->lowerbound[0]);
                }
                if (this->upperbound.size() != 0) {
                    this->set_upperbound(this->upperbound[0]);
                }

                this->prepare_problem();
                this->calc_optimal();
            }

            /** \fn set_number_of_variables(int number_of_variables)
             * 
             * Set dimension (`number_of_variables`) of the problem. When the 
             * `number_of_variables` is updated, `best_variables`, `lowerbound`, 
             * `upperbound`, and `optimal` need to be updated as well. In case 
             * the best value for each bit is not identical, another input 
             * 'best_variables' is provided.
             *
             * \param number_of_variables dimension 
             * \param best_variables bit values of the optimum
             */
            void set_number_of_variables(int number_of_variables,
                                         const std::vector<InputType> &
                                         best_variables                                        
                ) {
                this->number_of_variables = number_of_variables;
                this->best_variables = best_variables;
                if (this->lowerbound.size() != 0) {
                    this->set_lowerbound(this->lowerbound[0]);
                }
                if (this->upperbound.size() != 0) {
                    this->set_upperbound(this->upperbound[0]);
                }
                this->prepare_problem();
                this->calc_optimal();
            }

            /** \fn int get_number_of_objectives()
             * \brief Return number of objectives
             */
            int get_number_of_objectives() const {
                return this->number_of_objectives;
            }

            /** void set_number_of_objectives(int number_of_objectives)
             * \brief Set the number of objectives of the problem.
             * 
             * After setting the number of objectives, `raw_objectives`, 
             * `transformed_objectives`, `best_so_far_raw_objectives`,
             * and `best_so_far_transformed_objectives` will be allocated.
             * 
             * \param number_of_objectives number of objectives
             */
            void set_number_of_objectives(int number_of_objectives) {
                this->number_of_objectives = number_of_objectives;
                this->raw_objectives = std::vector<double>(this->number_of_objectives);
                this->transformed_objectives = std::vector<double>(this->number_of_objectives);
                if (this->maximization_minimization_flag == common::OptimizationType::maximization) {
                    this->best_so_far_raw_objectives = std::vector<double>(this->number_of_objectives,
                                                                           std::numeric_limits<double>::lowest());
                    this->best_so_far_transformed_objectives = std::vector<double>(
                        this->number_of_objectives, std::numeric_limits<double>::lowest());
                } else {
                    this->best_so_far_raw_objectives = std::vector<double>(this->number_of_objectives,
                                                                           std::numeric_limits<double>::max());
                    this->best_so_far_transformed_objectives = std::vector<double>(
                        this->number_of_objectives, std::numeric_limits<double>::max());
                }
                this->optimal = std::vector<double>(this->number_of_objectives);
            }

            /** \fn std::vector<double> get_raw_objectives()
             * \brief Return objective values before applying transformation on objectives.
             */
            std::vector<double> get_raw_objectives() const {
                return this->raw_objectives;
            }

            /** \fn std::vector<double> get_transformed_objectives()
             * \brief Return objective values after applying transformation on objectives.
             */
            std::vector<double> get_transformed_objectives() const {
                return this->transformed_objectives;
            }

            /** \fn int get_transformed_number_of_variables()
             * \brief Return dimension of the problem values after applying transformation 
             * on input variables.
             */
            int get_transformed_number_of_variables() const {
                return this->transformed_number_of_variables;
            }

            /** \fn std::vector<InputType> get_transformed_variables()
             * \brief Return transformed input variables.
             */
            std::vector<InputType> get_transformed_variables() const {
                return this->transformed_variables;
            }

            /** \fn std::vector<InputType> get_best_variables()
             * \brief Return optimal variables.
             */
            std::vector<InputType> get_best_variables() const {
                return this->best_variables;
            }

            /** void set_best_variables(InputType best_variables)
             * \brief Set `best_variables` of the problem with the given variable.
             * 
             * With this function, values at every index will be identical with the given value.
             * \param best_variable best variables
             */
            void set_best_variables(InputType best_variables) {
                this->best_variables.clear();
                for (auto i = 0; i < this->number_of_variables; ++i) {
                    this->best_variables.push_back(best_variables);
                }
            }

            /** void set_best_variables(const std::vector<InputType>& best_variables)
             * \brief Set `best_variables` of the problem with the given variables.
             * 
             * \param best_variable best variables
             */
            void set_best_variables(
                const std::vector<InputType> &best_variables) {
                this->best_variables = best_variables;
            }

            /** bool has_optimal()
             * \brief Detect if optimum of the problem is assigned/known.
             */
            bool has_optimal() const {
                return static_cast<int>(this->optimal.size()) == this->get_number_of_objectives();
            }

            /** std::vector<double> get_optimal()
             * \brief Return optimum if it is assigned/known.
             */
            std::vector<double> get_optimal() const {
                // FIXME unsure if one want to raise an exception in Release mode also?
                // Assert that the optimum have been initialized.
                assert(this->has_optimal());
                return this->optimal;
            }

            /** void set_optimal(double optimal)
             * \brief Set `optimal` of the problem with the given value.
             * 
             * With the function, all objectives of the optimum will be identical with the given value.
             * \param optimal optimal value
             */
            void set_optimal(double optimal) {
                std::vector<double>().swap(this->optimal);
                this->optimal.reserve(this->number_of_objectives);
                for (auto i = 0; i < this->number_of_objectives; ++i) {
                    this->optimal.push_back(optimal);
                }
            }

            /** void set_optimal(const std::vector<double>& optimal)
             * \brief Set `optimal` of the problem with the given values.
             * 
             * \param optimal optimal value
             */
            void set_optimal(const std::vector<double> &optimal) {
                this->optimal = optimal;
            }

            /** void evaluate_optimal()
             * \brief Calculate `optimal` of the problem with given variables.
             * 
             * \param best_variables best variables
             */
            void evaluate_optimal(std::vector<InputType> best_variables) {
                this->optimal[0] = this->evaluate(best_variables);
            }

            /** void evaluate_optimal()
             * \brief Calculate `optimal` of the problem.
             */
            void evaluate_optimal() {
                this->optimal[0] = this->evaluate(this->best_variables);
            }

            /** int get_evaluations()
             * \brief Return the evaluation times that has been done.
             */
            int get_evaluations() const {
                return this->evaluations;
            }

            /** std::vector<double> get_best_so_far_raw_objectives()
             * \brief Return objective values before applying transformation on objectives 
             * of the best-so-far solution.
             */
            std::vector<double> get_best_so_far_raw_objectives() const {
                return this->best_so_far_raw_objectives;
            }

            /** int get_best_so_far_raw_evaluations()
             * \brief Return evaluation times that the best-so-far raw objective was found.
             */
            int get_best_so_far_raw_evaluations() const {
                return this->best_so_far_raw_evaluations;
            }

            /** std::vector<double> get_best_so_far_transformed_objectives()
             * \brief Return objective values after applying transformation on objectives of
             *  the best-so-far solution.
             */
            std::vector<double> get_best_so_far_transformed_objectives() const {
                return this->best_so_far_transformed_objectives;
            }

            /** int get_best_so_far_transformed_evaluations()
             * \brief Return evaluation times that the best-so-far transformed objective 
             * was found.
             */
            int get_best_so_far_transformed_evaluations() const {
                return this->best_so_far_transformed_evaluations;
            }

            /** common::optimization_type get_optimization_type()
             * \brief Return the optimization type: maximization or minimizations.
             */
            common::OptimizationType get_optimization_type() const {
                return this->maximization_minimization_flag;
            }

            /** void set_as_maximization()
             * \brief Set the problem as maximization.
             */
            void set_as_maximization() {
                this->maximization_minimization_flag =
                    common::OptimizationType::maximization;
                for (std::size_t i = 0; i != this->number_of_objectives; ++i) {
                    this->best_so_far_raw_objectives[i] = std::numeric_limits<
                        double>::lowest();
                    this->best_so_far_transformed_objectives[i] =
                        std::numeric_limits<double>::lowest();
                }
            }

            /** void set_as_minimization()
             * \brief Set the problem as minimization.
             */
            void set_as_minimization() {
                this->maximization_minimization_flag =
                    common::OptimizationType::minimization;
                for (auto i = 0; i != this->number_of_objectives; ++i) {
                    this->best_so_far_raw_objectives[i] = std::numeric_limits<
                        double>::max();
                    this->best_so_far_transformed_objectives[i] =
                        std::numeric_limits<double>::max();
                }
            }


            friend std::ostream &operator<<(std::ostream &os, const base &obj) {
                return os << "f"
                       << obj.get_problem_id() << "_d"
                       << obj.get_number_of_variables() << "_i"
                       << obj.get_instance_id();
            }
        };
    } // namespace problem
}     // namespace ioh
