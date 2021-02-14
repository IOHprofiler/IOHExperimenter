#pragma once

#include "base.hpp"
#include "observer.hpp"
#include "ioh/common.hpp"
#include "ioh/experiment.hpp"

namespace ioh::logger {

    template <typename ProblemType>
    class Default : public Base<ProblemType>, public Observer {

        common::file::UniqueFolder experiment_folder_;
        ProblemType const *problem_{};
        std::string last_logged_line_;

        class InfoFile {
            friend class Default;

            std::string algorithm_name_;
            std::string algorithm_info_;
            std::string suite_name_ = "No suite";
            common::file::BufferedFileStream stream_;

            /**
            * \brief List of attributes constant per experiment
            */
            std::map<std::string, std::string> experiment_attributes_;

            /**
             * \brief List of parameters constant per run
             */
            std::map<std::string, std::shared_ptr<double>> run_attributes_;

            struct {
                double y, transformed_y;
                size_t evaluations;
            } best_point_;

        public:
            InfoFile(std::string algorithm_name, std::string algorithm_info)
                : algorithm_name_(std::move(algorithm_name)),
                  algorithm_info_(std::move(algorithm_info)) {
            }

            static char maximization_flag(const common::OptimizationType optimization_type) {
                return optimization_type == common::OptimizationType::maximization ? 'T' : 'F';
            }

            void update_run_info(ProblemType const *current_problem) {
                if (current_problem != nullptr) {
                    stream_ << common::string_format(", %d:%d|%g", current_problem->get_instance_id(),
                                                     best_point_.evaluations, best_point_.y);

                    for (auto& [key, value] : run_attributes_) {
                        stream_ << common::string_format(";%g", *value);
                    }
                        
                }
            }

            void track_problem(const ProblemType &problem, ProblemType const *current_problem,
                               const fs::path &root_path) {
                using namespace common;

                const int id = problem.get_problem_id();
                const std::string name = problem.get_problem_name();
                const int d = problem.get_number_of_variables();

                update_run_info(current_problem);

                if (current_problem == nullptr || id != current_problem->get_problem_id())
                    stream_ = file::BufferedFileStream(
                        root_path, string_format("IOHprofiler_f%d_%s.info", id, name.c_str()));

                if (current_problem == nullptr || d != current_problem->get_number_of_variables() || id !=
                    current_problem->get_problem_id()) {
                    stream_.write(
                        string_format(
                            R"(suite = "%s", funcId = %d, funcName = "%s", DIM = %d, maximization = "%c", algId = "%s", algInfo = "%s")",
                            suite_name_.c_str(), id, name.c_str(), d,
                            maximization_flag(problem.get_optimization_type()),
                            algorithm_name_.c_str(), algorithm_info_.c_str()),
                        true);

                    // extra attrs const per experiment
                    for (const auto &[fst, snd] : experiment_attributes_)
                        stream_ << string_format(R"(, %s = "%s")", fst.c_str(), snd.c_str());

                    // extra attrs per run
                    if (!run_attributes_.empty()) {
                        stream_ << ", dynamicAttribute = \"";
                        for (auto i = run_attributes_.begin(); i != run_attributes_.end();)
                            stream_ << i->first << (++i != run_attributes_.end() ? "|" : "");
                        stream_ << "\"";
                    }

                    stream_ << "\n%\n" << string_format(
                        "data_f%d_%s/IOHprofiler_f%d_DIM%d.dat", id, name.c_str(), id, d);
                }
            }

        } info_file_;

        class DataFiles {
            friend class Default;

            struct File {
                bool active = false;
                bool triggered = false;
                common::file::BufferedFileStream stream;
                std::string extension;

                File(const bool act, std::string ext, const bool trig = false)
                    : active(act), triggered(trig), extension(std::move(ext)) {
                }

                ~File() {
                    stream.close();
                }
            };

            std::array<File, 4> files_;

            /**
              * \brief List of additional parameters to be logged at each trigger
              */
            std::map<std::string, std::shared_ptr<double>> logged_attributes_;

            [[nodiscard]] std::string header() {
                std::string header =
                    R"#("function evaluation" "current f(x)" "best-so-far f(x)" "current af(x)+b" "best af(x)+b")#";
                for (const auto &[fst, snd] : logged_attributes_)
                    header += common::string_format(R"( "%s")", fst.c_str());
                return header + "\n";
            }

        public:
            DataFiles(const bool d_active, const bool c_active, const bool t_active, const bool i_active)
                : files_{File(d_active, ""), File(t_active, "t"), File(i_active, "i"), File(c_active, "c", c_active)} {

            }

            /**
             * \brief Sets the values for the triggers
             * \param triggers array of triggers, order=(.dat, .tdat, idat)
             */
            void update_triggers(const std::array<bool, 3> triggers) {
                for (size_t i = 0; i < triggers.size(); i++)
                    files_[i].triggered = triggers[i];
            }

            void track_problem(const ProblemType &problem, const fs::path &root) {
                const int id = problem.get_problem_id();
                const std::string name = problem.get_problem_name();
                const int d = problem.get_number_of_variables();

                const auto directory = root / common::string_format("data_f%d_%s", id, name.c_str());

                for (auto &file : files_) {
                    if (file.active) {
                        if (!exists(directory))
                            create_directory(directory);
                        auto filename = common::string_format("IOHprofiler_f%d_DIM%d.%sdat", id, d,
                                                              file.extension.c_str());
                        file.stream = common::file::BufferedFileStream(directory, filename);
                        file.stream << header();
                    }
                }
            }

            void write(const std::string &data, const bool all = false) {
                for (auto &file : files_)
                    if ((!all && file.triggered) || (all && (file.active && !file.triggered)))
                        file.stream << data;
            }

            DataFiles &operator<<(const std::string &data) {
                write(data);
                return *this;
            }

        } data_files_;


    public:
        explicit Default(fs::path output_directory,
                         const std::string &folder_name = "ioh_data",
                         std::string algorithm_name = "algorithm_name",
                         std::string algorithm_info = "algorithm_info",
                         const bool t_always = false,
                         const int t_on_interval = 0,
                         const int t_per_time_range = 0,
                         const bool t_on_improvement = true,
                         const std::vector<int> &t_at_time_points = {0},
                         const common::OptimizationType optimization_type = common::OptimizationType::minimization,
                         const int trigger_at_time_points_exp_base = 10,
                         const int trigger_at_range_exp_base = 10
            )
            : Observer(t_always, t_on_interval, t_per_time_range, t_on_improvement, t_at_time_points, optimization_type,
                       trigger_at_time_points_exp_base, trigger_at_range_exp_base),
              experiment_folder_(std::move(output_directory), folder_name),
              info_file_(std::move(algorithm_name), std::move(algorithm_info)),
              data_files_(trigger_on_improvement(), trigger_always(),
                          trigger_at_time_points() || trigger_at_time_range(),
                          trigger_at_interval()) {
        }

        explicit Default(const std::string &folder_name = "ioh_data",
                         const std::string &algorithm_name = "algorithm_name",
                         const std::string &algorithm_info = "algorithm_info",
                         const bool t_always = false,
                         const int t_on_interval = 0,
                         const int t_per_time_range = 0,
                         const bool t_on_improvement = true,
                         const std::vector<int> &t_at_time_points = {0},
                         const common::OptimizationType optimization_type = common::OptimizationType::minimization,
                         const int trigger_at_time_points_exp_base = 10,
                         const int trigger_at_range_exp_base = 10
            )
            : Default(fs::current_path(), folder_name, algorithm_name, algorithm_info,
                      t_always, t_on_interval, t_per_time_range, t_on_improvement, t_at_time_points,
                      optimization_type, trigger_at_time_points_exp_base, trigger_at_range_exp_base) {
        }

        explicit Default(const bool t_always,
                         const int t_on_interval = 0,
                         const int t_per_time_range = 0,
                         const bool t_on_improvement = true,
                         const std::vector<int> &t_at_time_points = {0},
                         const common::OptimizationType optimization_type = common::OptimizationType::minimization,
                         const int trigger_at_time_points_exp_base = 10,
                         const int trigger_at_range_exp_base = 10
            )
            : Default("ioh_data", "algorithm_name", "algorithm_info", t_always, t_on_interval, t_per_time_range,
                      t_on_improvement, t_at_time_points, optimization_type, trigger_at_time_points_exp_base,
                      trigger_at_range_exp_base) {
        }

        explicit Default(const experiment::Configuration &conf)
            : Default(conf.get_result_folder(), conf.get_output_directory(), conf.get_algorithm_name(),
                      conf.get_algorithm_info(), conf.get_complete_triggers(), conf.get_number_interval_triggers(),
                      conf.get_number_target_triggers(), conf.get_update_triggers(), conf.get_base_evaluation_triggers()
                ) {
        }


        ~Default() {
            info_file_.update_run_info(problem_); // needs to call on close
            data_files_.write(last_logged_line_, true);
        }

        void track_problem(const ProblemType &problem) override {
            data_files_.write(last_logged_line_, true);

            info_file_.track_problem(problem, problem_, experiment_folder_.path());
            data_files_.track_problem(problem, experiment_folder_.path());

            problem_ = &problem;
            reset(problem.get_optimization_type());

            if (problem.get_evaluations() != 0) // problem is not yet reset
                const_cast<ProblemType &>(problem).reset_problem();

        }

        void track_suite(const suite::base<ProblemType> &suite) override {
            info_file_.suite_name_ = suite.get_suite_name();
        }

        void do_log(const std::vector<double> &log_info) override {
            const size_t evaluations = log_info[0];
            const auto y = log_info[1];
            const auto y_best = log_info[2];
            const auto transformed_y = log_info[3];
            const auto transformed_y_best = log_info[4];

            const bool improvement_found = improvement_trigger(transformed_y, problem_->get_optimization_type());

            data_files_.update_triggers({
                improvement_found,
                time_points_trigger(evaluations) || time_range_trigger(evaluations),
                interval_trigger(evaluations)
            });

            // We can do this with pointers, more efficient
            last_logged_line_ = common::string_format("%d %f %f %f %f", evaluations, y, y_best, transformed_y,  
                                                      transformed_y_best);
            for (const auto &e : data_files_.logged_attributes_)
                last_logged_line_ += common::string_format(" %f", *e.second);

            last_logged_line_ += "\n";

            data_files_ << last_logged_line_;

            if (improvement_found)
                info_file_.best_point_ = {y, transformed_y, evaluations};
        }

        [[nodiscard]] common::file::UniqueFolder &experiment_folder() {
            return experiment_folder_;
        }


        /// Parameters ///
        /// We have:
        ///     experiment attributes -> only once per experiment (info file)
        ///     run attributes -> (can) change per run (info file)
        ///     logged attributes -> logged at every log-moment (data files)
        /// TODO: standardize naming of the methods for accessing these
        template <typename V>
        void add_experiment_attribute(const std::string name, const V value) {
            info_file_.experiment_attributes_[name] = common::to_string(value);
        }

        void delete_experiment_attribute(const std::string name) {
            info_file_.experiment_attributes_.erase(name);
        }

        void create_run_attributes(const std::vector<std::shared_ptr<double>> &attributes,
                                   const std::vector<std::string> &attribute_names,
                                   const bool clear = true) {
            if (attribute_names.size() != attributes.size())
                common::log::error("Attributes and their names are given with different size.");

            if (clear && !info_file_.run_attributes_.empty())
                info_file_.run_attributes_.clear();

            for (size_t i = 0; i != attributes.size(); i++)
                info_file_.run_attributes_[attribute_names[i]] = attributes[i];
        }

        void create_run_attributes(const std::vector<std::shared_ptr<double>> &attributes) {
            std::vector<std::string> names(attributes.size());
            std::generate(names.begin(), names.end(), [n = 1]() mutable
            {
                return common::string_format("attr%d", n++);
            });

            create_run_attributes(attributes, names);
        }

        void create_run_attributes(const std::vector<std::string> &attribute_names) {
            std::vector<std::shared_ptr<double>> attributes(attribute_names.size());
            std::generate(attributes.begin(), attributes.end(), []()
            {
                return std::make_shared<double>(-9999);
            });
            create_run_attributes(attributes, attribute_names);
        }


        void add_run_attributes(const std::vector<std::string> &attributes_name,
                                const std::vector<double> &attributes) {
            std::vector<std::shared_ptr<double>> ptr_attributes(attributes.size());
            std::generate(ptr_attributes.begin(), ptr_attributes.end(), [n=0, attributes]() mutable
            {
                return std::make_shared<double>(attributes[n++]);
            });
            create_run_attributes(ptr_attributes, attributes_name, false);
        }

        void set_run_attributes(const std::vector<std::string> &attribute_names,
                                const std::vector<double> &attributes) {
            if (attribute_names.size() != attributes.size())
                common::log::error("Attributes and their names are given with different size.");

            for (size_t i = 0; i != attribute_names.size(); ++i)
                if (info_file_.run_attributes_.find(attribute_names[i]) != info_file_.run_attributes_.end())
                    *info_file_.run_attributes_[attribute_names[i]] = attributes[i];
                else
                    common::log::error("Dynamic attribute " + attribute_names[i] + " does not exist");
        }


        void create_logged_attributes(const std::vector<std::string> &attribute_names) {
            if (!data_files_.logged_attributes_.empty())
                data_files_.logged_attributes_.clear();

            for (size_t i = 0; i != attribute_names.size(); i++)
                data_files_.logged_attributes_[attribute_names[i]] = std::make_shared<double>(-9999);
        }

        void create_logged_attributes(const std::vector<std::string> &attribute_names,
                                      const std::vector<std::shared_ptr<double>> &attributes) {
            if (attribute_names.size() != attributes.size())
                common::log::error("Attributes and their names are given with different size.");

            for (size_t i = 0; i != attribute_names.size(); i++)
                data_files_.logged_attributes_[attribute_names[i]] = attributes[i];

        }


        void create_logged_attributes(const std::vector<std::string> &attribute_names,
                                      const std::vector<double> &attributes) {
            if (attribute_names.size() != attributes.size())
                common::log::error("Attributes and their names are given with different size.");

            for (size_t i = 0; i != attribute_names.size(); i++)
                data_files_.logged_attributes_[attribute_names[i]] = std::make_shared<double>(attributes[i]);

        }

        void set_logged_attributes(const std::vector<std::string> &attribute_names,
                                   const std::vector<double> &attributes) {
            if (attribute_names.size() != attributes.size())
                common::log::error("Attributes and their names are given with different size.");

            for (size_t i = 0; i != attribute_names.size(); ++i)
                if (data_files_.logged_attributes_.find(attribute_names[i]) != data_files_.logged_attributes_.end())
                    *data_files_.logged_attributes_[attribute_names[i]] = attributes[i];
                else
                    common::log::error("Attribute " + attribute_names[i] + " does not exist");
        }
    };
}
