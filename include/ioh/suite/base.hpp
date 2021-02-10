#pragma once

#include "ioh/common/class_generator.hpp"
#include "ioh/problem/base.hpp"

namespace ioh {
    namespace suite {
        class abstract_suite {
        public:
            abstract_suite() = default;
        };


        /// \brief A base class for construct a suite for sets of problem::bases.
        ///
        /// To specify available problems of a suite, registerProblem must be implemented in derived class.
        /// The default lable of problems are string type. Integer type are also optional, but we highly
        /// recommond registering problem with string lable and creating a map of string problem_name and integer problem_id.
        template <typename ProblemType>
        class base
            :
            // : public ProblemType,
            // public std::vector<std::shared_ptr<ProblemType>>,
            public abstract_suite {

            // std::vector<ProblemType> problems_;
            std::vector<std::shared_ptr<ProblemType>> problems_;

        public:
            typedef std::shared_ptr<ProblemType> problem_ptr;

            base(std::vector<int> problem_id = std::vector<int>(0),
                 std::vector<int> instance_id = std::vector<int>(0),
                 std::vector<int> dimension = std::vector<int>(0))
                : suite_name("no suite"),
                  problem_id(problem_id),
                  instance_id(instance_id),
                  dimension(dimension),
                  problem_list_index(0),
                  size_of_problem_list(0),
                  get_problem_flag(false),
                  load_problem_flag(false),
                  current_problem(nullptr) {

            }

            // base(const base &) = delete;
            // base &operator =(const base &) = delete;

            /// \fn virtual void registerProblem()
            /// \brief A virtual function for registering problems.
            ///
            /// This function implements interfaces of available problems of a suite. With those interface,
            /// user are able to request problem together with problem_id, instance_id, and dimension.
            virtual void register_problems() = 0;


            /// \fn loadProblems()
            /// \brief Allocating memory and creating instances of problems to be included in the suite.
            ///
            /// Before acquiring a problem from the suite, this function must be invoked.
            /// Otherwise the list of problem is empty.
            void load_problem() {
                if (!this->problems_.empty()) {
                    this->problems_.clear();
                }
                this->size_of_problem_list =
                    this->number_of_dimensions * this->number_of_instances *
                    this->
                    number_of_problems;
                this->problem_list_index = 0;

                for (auto i = 0; i != this->number_of_problems; ++i) {
                    for (auto j = 0; j != this->number_of_dimensions; ++j) {
                        for (auto h = 0; h != this->number_of_instances; ++h) {
                            problem_ptr p = get_problem(
                                this->problem_id_name_map[this->problem_id[i]],
                                this->instance_id[h],
                                this->dimension[j]);
                            this->problems_.push_back(p);
                        }
                    }
                }
                assert(this->size_of_problem_list == this->problems_.size());
                this->get_problem_flag = false;
                this->load_problem_flag = true;
            }

            /// \fn std::shared_ptr<problem::base<T>> get_next_problem()
            /// \brief An interface of requesting problems in suite.
            ///
            /// To request 'the next' problem in the suite of corresponding problem_id, instance_id and dimension index.
            problem_ptr get_next_problem() {
                if (this->load_problem_flag == false)
                    this->load_problem();

                if (this->size_of_problem_list == 0) {
                    common::log::warning("There is no problem in the suite");
                    return nullptr;
                }

                if (this->problem_list_index == this->size_of_problem_list - 1
                    && this->
                    get_problem_flag == true)
                    return nullptr;

                if (this->get_problem_flag == false)
                    this->get_problem_flag = true;
                else
                    ++this->problem_list_index;

                this->current_problem = problems_.at(problem_list_index);

                this->current_problem->reset_problem();
                return this->current_problem;
            }

            /// \fn std::shared_ptr<problem::base<T>> get_current_problem()
            /// \brief An interface of requesting problems in suite.
            ///
            /// To request 'the current' problem in the suite of correponding problem_id, instance_id and dimension index.
            problem_ptr get_current_problem() {
                if (this->load_problem_flag == false)
                    this->load_problem();

                if (this->get_problem_flag == false)
                    this->get_problem_flag = true;

                // TODO: this may introduce unwanted behavior				
                this->current_problem = problems_.at(problem_list_index);
                this->current_problem->reset_problem();
                return this->current_problem;
            }

            template <typename P, typename T>
            void register_problem(const std::string name, const int id) {
                common::RegisterInFactory<P, T, int, int> problem(name);
                this->mapIDTOName(id, name);
            }

            void check_parameter_bounds(std::vector<int> ids, const int lb,
                                        const int ub) {
                for (const auto &e : ids)
                    if (e < lb || e > ub)
                        common::log::error(
                            "problem_id " + std::to_string(e) + " is not in " +
                            get_suite_name());
            }

            /// \fn Problem_ptr get_next_problem()
            /// \brief An interface of requesting problems in suite.
            ///
            /// To request a specific problem with corresponding problem_id, instance_id and dimension,
            /// without concerning the order of testing problems.
            problem_ptr get_problem(std::string problem_name, int instance,
                                    int dimension) {
                problem_ptr p = common::Factory<ProblemType, int, int>::get().
                    create(
                        problem_name, instance, dimension);
                assert(p != nullptr);
                return p;
            }

            /// \fn Problem_ptr get_next_problem()
            /// \brief An interface of requesting problems in suite.
            ///
            /// To request a specific problem with corresponding problem_id, instance_id and dimension,
            /// without concerning the order of testing problems.
            problem_ptr get_problem(int problem_id, int instance,
                                    int dimension) {
                auto problem_name = this->problem_id_name_map[problem_id];
                return get_problem(problem_name, instance, dimension);
            }

            int get_number_of_problems() const {
                return this->number_of_problems;
            }




            int get_number_of_instances() const {
                return this->number_of_instances;
            }


            int get_number_of_dimensions() const {
                return this->number_of_dimensions;
            }


            std::vector<int> get_problem_id() const {
                return this->problem_id;
            }

            std::map<int, std::string> get_problem_name() const {
                return this->problem_id_name_map;
            }

            //! Return a map of problem name to problem ID, for all the registered problem.
            std::map<std::string, int> get_problems() const {
                return this->problem_name_id_map;
            }

            std::vector<int> get_instance_id() const {
                return this->instance_id;
            }

            std::vector<int> get_dimension() const {
                return this->dimension;
            }

            std::string get_suite_name() const {
                return this->suite_name;
            }

            void set_suite_problem_id(const std::vector<int> &problem_id) {
                common::copy_vector(problem_id, this->problem_id);
                this->number_of_problems = static_cast<int>(this->problem_id.size());
            }

            void set_suite_instance_id(const std::vector<int> &instance_id) {
                common::copy_vector(instance_id, this->instance_id);
                this->number_of_instances = static_cast<int>(this->instance_id.size());
            }

            void set_suite_dimension(const std::vector<int> &dimension) {
                common::copy_vector(dimension, this->dimension);
                this->number_of_dimensions = static_cast<int>(this->dimension.size());
            }


            void set_suite_name(std::string suite_name) {
                this->suite_name = suite_name;
            }

            void mapIDTOName(const int id, const std::string &name) {
                problem_id_name_map[id] = name;
                problem_name_id_map[name] = id;
            }

            [[nodiscard]]
            std::vector<std::shared_ptr<ProblemType>> problems() const {
                return problems_;
            }

        private:
            std::string suite_name;
            int number_of_problems;
            int number_of_instances;
            int number_of_dimensions;

            std::vector<int> problem_id;
            std::vector<int> instance_id;
            std::vector<int> dimension;

            std::map<int, std::string> problem_id_name_map;
            std::map<std::string, int> problem_name_id_map;

            size_t problem_list_index;
            size_t size_of_problem_list;
            bool get_problem_flag;
            bool load_problem_flag;

            problem_ptr current_problem;
        };
    }
}