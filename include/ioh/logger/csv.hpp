#pragma once

#include "base.hpp"
#include "observer.hpp"
#include "ioh/common.hpp"
#include "ioh/experiment.hpp"

#include <array>

namespace ioh {
    namespace logger {
        template <class T>
        class Csv : public Base<T>, public Observer {
            // The information for directory.
            std::string folder_name;
            fs::path output_directory;

            // The information for logging.
            std::string algorithm_name;
            std::string algorithm_info;
            common::OptimizationType maximization_minimization_flag = common::OptimizationType::maximization;
            std::map<std::string, std::string> attr_per_exp_name_value;
            std::map<std::string, std::shared_ptr<double>> attr_per_run_name_value;

            std::string suite_name = "No suite";

            int dimension;
            int problem_id;
            int instance;
            std::string problem_name;

            std::vector<double> best_y;
            std::vector<double> best_transformed_y;
            size_t optimal_evaluations;
            std::vector<double> last_y;
            std::vector<double> last_transformed_y;
            size_t last_evaluations;

            std::map<std::string, std::shared_ptr<double>> logging_parameters;

            // fstream
            std::fstream cdat;
            std::fstream idat;
            std::fstream dat;
            std::fstream tdat;
            std::fstream info_file;

            std::string cdat_buffer;
            std::string idat_buffer;
            std::string dat_buffer;
            std::string tdat_buffer;
            std::string info_buffer;


            int last_dimension = 0;
            int last_problem_id = -1;

            bool header_flag;
            /// < parameters to track if the header line is logged.


            /// \fn std::string experiment_folder_name()
            /// \brief return an available name of folder to be created.
            ///
            /// To create a name of the folder logging files.
            /// If there exists a files or a folder with the same name, the expected 
            /// directory will be renamed by adding a suffix.
            /// For example,
            ///     If a folder or file 'test' has already been in correct path, the 
            ///     expected directory will be renamed as 'test-1', 'test-2', ... 
            ///     until there is no such a folder or file. 
            fs::path experiment_folder_name() {
                auto renamed_directory = output_directory / folder_name;
                auto temp_folder_name = folder_name;
                auto index = 0;
                while (exists(renamed_directory)) {
                    ++index;
                    temp_folder_name = folder_name + '-' + common::to_string(index);
                    renamed_directory = output_directory / temp_folder_name;
                }
                folder_name = temp_folder_name;
                return renamed_directory;
            }

            void create_folder(const fs::path path) const {
                try {
                    create_directories(path);
                } catch (const std::exception &e) {
                    common::log::error(
                        std::string("Error on creating directory:") + e.what());
                }
            }

            std::string dat_header() const {
                std::string dat_header =
                    "\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\"  \"best af(x)+b\"";

                if (logging_parameters.size() != 0)
                    for (auto const &e : logging_parameters)
                        dat_header += " \"" + e.first + "\"";
                return dat_header;
            }


            void recreate_handle(const bool execute, const std::string &ext,
                                 std::fstream &handle,
                                 std::string &buffer) const {
                if (execute) {
                    const auto file_path = get_file_path(ext);

                    if (handle.is_open())
                        handle.close();

                    handle.open(file_path.c_str(),
                                std::ofstream::out | std::ofstream::app);
                    buffer = "";
                    const auto header = dat_header();

                    write_in_buffer(header + "\n", buffer, handle);
                }
            }

            static void write_handle(std::fstream &handle, std::string &buffer) {
                if (handle.is_open()) {
                    write_stream(buffer, handle);
                    handle.flush();
                    buffer.clear();
                }
            }

            void write_handle(std::fstream &handle, std::string &buffer,
                              std::string &line) const {
                if (handle.is_open()) {
                    write_in_buffer(line, buffer, handle);
                    write_handle(handle, buffer);
                }
            }

            void write_header() {
                const auto sub_dir = sub_directory();

                if (!exists(sub_dir))
                    create_folder(sub_dir);

                recreate_handle(this->trigger_always(), ".cdat", cdat, cdat_buffer);
                recreate_handle(this->trigger_at_interval(), ".idat", idat, idat_buffer);
                recreate_handle(this->trigger_on_improvement(), ".dat", dat, dat_buffer);
                recreate_handle(this->trigger_at_time_points() || this->trigger_at_time_range(), ".tdat", tdat,
                                tdat_buffer);
            }

            static void write_stream(std::string buffer_string,
                                     std::fstream &dat_stream) {
                dat_stream.write(buffer_string.c_str(),
                                 sizeof(char) * buffer_string.size());
            }

            static void write_in_buffer(std::string add_string,
                                        std::string &buffer_string,
                                        std::fstream &dat_stream) {
                if (!dat_stream.is_open())
                    common::log::error("file is not open");

                if (buffer_string.size() + add_string.size() < IOH_MAX_BUFFER_SIZE) {
                    buffer_string = buffer_string + add_string;
                } else {
                    write_stream(buffer_string, dat_stream);
                    buffer_string.clear();
                    buffer_string = add_string;
                }
            }

            static void conditional_write_in_buffer(const bool write,
                                                    std::string add_string,
                                                    std::string &buffer_string,
                                                    std::fstream &dat_stream) {
                if (write)
                    write_in_buffer(add_string, buffer_string, dat_stream);
            }

            /// \fn openIndex()
            /// \brief to create the folder of logging files.
            void open_index() {
                create_folder(experiment_folder_name());
            }

        public:

            explicit Csv(const std::string output_directory = "./",
                         const std::string folder_name = "IOHprofiler_test",
                         const std::string algorithm_name = "algorithm",
                         const std::string algorithm_info = "algorithm_info")
                : folder_name{folder_name},
                  output_directory{output_directory},
                  algorithm_name{algorithm_name},
                  algorithm_info{algorithm_info} {
                open_index();
            }

            /**
             * \brief Used in experimenter, shorthand constructor for instantiating loggers from
             * configuration objects.
             * \param conf a \ref experiment::Configuration object
             */
            explicit Csv(const experiment::Configuration &conf)
                : Observer(conf.get_complete_triggers(),
                           conf.get_number_interval_triggers(),
                           conf.get_number_target_triggers(),
                           conf.get_update_triggers(),
                           conf.get_base_evaluation_triggers()),
                  folder_name{conf.get_result_folder()},
                  output_directory{conf.get_output_directory()},
                  algorithm_name{conf.get_algorithm_name()},
                  algorithm_info{conf.get_algorithm_info()} {
                open_index();
            }

            ~Csv() {
                clear_logger();
            }

            [[nodiscard]] fs::path experiment_directory() const {
                return absolute(output_directory / folder_name);
            }

            [[nodiscard]] fs::path sub_directory() const {
                return experiment_directory() / ("data_f" + common::to_string(problem_id) + "_" + problem_name);
            }

            [[nodiscard]] fs::path get_file_path(const std::string &ext) const {
                return sub_directory() /
                       ("IOHprofiler_f" + common::to_string(problem_id)
                        + "_DIM" + common::to_string(dimension) + ext);
            }

            [[nodiscard]] fs::path get_info_file_path() const {
                return experiment_directory() /
                       ("IOHprofiler_f" + common::to_string(problem_id) + "_" +
                        problem_name + ".info");
            }

            void clear_logger() {
                if (info_file.is_open()) {
                    write_info(this->instance, this->best_y[0],
                               this->best_transformed_y[0],
                               this->optimal_evaluations,
                               this->last_y[0], this->last_transformed_y[0],
                               this->last_evaluations);
                    info_file.close();
                }

                /// Close .dat files after .info file to record evaluations of the last iteration.
                if (cdat.is_open())
                    cdat.close();
                if (idat.is_open())
                    idat.close();
                if (dat.is_open())
                    dat.close();
                if (tdat.is_open())
                    tdat.close();
            }

            template <typename V>
            void add_attribute(const std::string name, const V value) {
                this->attr_per_exp_name_value[name] = common::to_string(value);
            }

            void delete_attribute(const std::string name) {
                this->attr_per_exp_name_value.erase(name);
            }

            /// \fn track_problem(int problem_id, int dimension, int get)
            ///
            /// This function is to be invoked by problem class.
            /// To update info of current working problem, and to write headline in corresponding files.

            void track_problem(int problem_id, int dimension, int instance,
                               std::string problem_name,
                               common::OptimizationType
                               maximization_minimization_flag) {
                /// Handle info of the previous problem.
                if (info_file.is_open())
                    write_info(instance, best_y[0], best_transformed_y[0],
                               optimal_evaluations,
                               last_y[0], last_transformed_y[0],
                               last_evaluations);

                optimal_evaluations = 0;
                last_evaluations = 0;

                /// TO DO: Update the method of initializing this value.
                best_y.clear();
                best_transformed_y.clear();
                last_y.clear();
                last_transformed_y.clear();
                if (maximization_minimization_flag == common::OptimizationType::maximization) {
                    best_y.push_back(-std::numeric_limits<double>::max());
                    best_transformed_y.push_back(-std::numeric_limits<double>::max());
                    last_y.push_back(-std::numeric_limits<double>::max());
                    last_transformed_y.push_back(-std::numeric_limits<double>::max());
                } else {
                    best_y.push_back(std::numeric_limits<double>::max());
                    best_transformed_y.push_back(std::numeric_limits<double>::max());
                    last_y.push_back(std::numeric_limits<double>::max());
                    last_transformed_y.push_back(std::numeric_limits<double>::max());
                }

                this->reset(maximization_minimization_flag);
                this->problem_id = problem_id;
                this->dimension = dimension;
                this->instance = instance;
                this->problem_name = problem_name;
                this->maximization_minimization_flag = maximization_minimization_flag;

                open_info();
                header_flag = false;
            }


            void track_problem(const T &problem) override {
                track_problem(
                    problem.get_problem_id(),
                    problem.get_number_of_variables(),
                    problem.get_instance_id(),
                    problem.get_problem_name(),
                    problem.get_optimization_type()
                    );
            }

            void track_suite(const std::string &suite_name) {
                this->suite_name = suite_name;
            }

            void track_suite(const suite::base<T> &suite) override {
                track_suite(suite.get_suite_name());
            }

            void open_info() {
                const auto optimization_type = std::array<std::string, 2>{"T", "F"}[0];

                if (problem_id != this->last_problem_id || dimension != this->last_dimension) {
                    if (this->info_file.is_open())
                        this->info_file << "\n";

                    this->info_file.close();
                    const auto infofile_path = output_directory / folder_name / common::string_format(
                                                   "IOHprofiler_f%d_%s.info", problem_id, problem_name.c_str());

                    this->info_file.open(infofile_path.c_str(),
                                         std::ofstream::out |
                                         std::ofstream::app);

                    this->info_file << common::string_format(
                        R"(suite = "%s", funcId = %d, funcName = "%s", DIM = %d, maximization = "%s", algId = "%s", algInfo = "%s")",
                        suite_name.c_str(),
                        problem_id,
                        problem_name.c_str(),
                        dimension,
                        optimization_type.c_str(),
                        algorithm_name.c_str(),
                        algorithm_info.c_str()
                        );

                    for (const auto &[fst, snd] : attr_per_exp_name_value)
                        this->info_file << common::string_format(R"(, %s = "%s")", fst.c_str(), snd.c_str());

                    if (!this->attr_per_run_name_value.empty()) {
                        this->info_file << ", dynamicAttribute = \"";

                        for (auto iter = this->
                                         attr_per_run_name_value.begin(); iter != this->attr_per_run_name_value.end();
                        ) {
                            this->info_file << iter->first;
                            if (++iter != this->attr_per_run_name_value.end()) {
                                this->info_file << "|";
                            }
                        }
                        this->info_file << "\"";
                    }

                    this->info_file << "\n%\n" + common::string_format(
                        R"(data_f%d_%s/IOHprofiler_f%d_DIM%d.dat)",
                        problem_id,
                        problem_name.c_str(),
                        problem_id,
                        dimension
                        );

                    this->last_problem_id = problem_id;
                    this->last_dimension = dimension;
                }
            }

            void write_info(const int instance, const double best_y,
                            const double best_transformed_y,
                            const size_t evaluations,
                            const double last_y,
                            const double last_transformed_y,
                            const size_t last_evaluations) {
                if (!info_file.is_open())
                    common::log::error(
                        "write_info(): writing info into unopened info_file");

                this->info_file << ", " + common::to_string(instance) + ":" +
                    common::to_string(evaluations) + "|" + common::to_string(best_y);

                common::string_format(", %d:%d|%f", instance, evaluations, best_y);

                if (!this->attr_per_run_name_value.empty()) {
                    this->info_file << ";";
                    for (auto iter = this->attr_per_run_name_value.
                                           begin(); iter != this->attr_per_run_name_value.end();) {
                        this->info_file << common::to_string(*iter->second);
                        if (++iter != this->attr_per_run_name_value.end()) {
                            this->info_file << "|";
                        }
                    }
                }

                if (evaluations != last_evaluations) // need write
                {
                    auto written_line =
                        common::to_string(last_evaluations) + " " + common::to_string(last_y) +
                        " " + common::to_string(best_y) + " " + common::to_string(last_transformed_y) + " "
                        + common::to_string(best_transformed_y);

                    for (auto iter = this->logging_parameters.begin();
                         iter != this->logging_parameters.end(); ++iter)
                        written_line += " " + common::to_string(*iter->second);

                    written_line += '\n';
                    this->write_handle(cdat, cdat_buffer, written_line);
                    this->write_handle(idat, idat_buffer, written_line);
                    this->write_handle(dat, dat_buffer, written_line);
                    this->write_handle(tdat, tdat_buffer, written_line);
                }

                this->write_handle(cdat, cdat_buffer);
                this->write_handle(idat, idat_buffer);
                this->write_handle(dat, dat_buffer);
                this->write_handle(tdat, tdat_buffer);
            }

            void do_log(const std::vector<double> &log_info) override {
                this->write_line(static_cast<size_t>(log_info[0]), log_info[1],
                                 log_info[2],
                                 log_info[3], log_info[4]);
            }

            void write_line(size_t evaluations, double y, double best_so_far_y,
                            double transformed_y,
                            double best_so_far_transformed_y) {
                if (header_flag == false) {
                    this->write_header();
                    header_flag = true;
                }

                this->last_evaluations = evaluations;
                this->last_y[0] = y;
                this->last_transformed_y[0] = transformed_y;

                const bool cdat_flag = this->trigger_always();
                const bool idat_flag = this->interval_trigger(evaluations);
                const bool dat_flag = this->improvement_trigger(transformed_y, maximization_minimization_flag);
                const bool tdat_flag = this->time_points_trigger(evaluations) || this->time_range_trigger(evaluations);

                const auto need_write =
                    cdat_flag || idat_flag || dat_flag || tdat_flag;

                if (need_write) {
                    auto written_line = common::to_string(evaluations) + " " +
                                        common::to_string(y) + " "
                                        + common::to_string(best_so_far_y) + " "
                                        +
                                        common::to_string(transformed_y) + " "
                                        + common::to_string(
                                            best_so_far_transformed_y);

                    if (this->logging_parameters.size() != 0) {
                        for (auto const &e : logging_parameters)
                            written_line += " " + common::to_string(*e.second);
                    }

                    written_line += '\n';
                    conditional_write_in_buffer(cdat_flag, written_line,
                                                cdat_buffer, cdat);
                    conditional_write_in_buffer(idat_flag, written_line,
                                                idat_buffer, idat);
                    conditional_write_in_buffer(dat_flag, written_line,
                                                dat_buffer, dat);
                    conditional_write_in_buffer(tdat_flag, written_line,
                                                tdat_buffer, tdat);
                }

                if (common::compare_objectives(transformed_y, this->best_transformed_y[0],
                                               this->maximization_minimization_flag))
                    this->update_logger_info(evaluations, y, transformed_y);
            }


            void update_logger_info(size_t optimal_evaluations, double y,
                                    double transformed_y) {
                this->optimal_evaluations = optimal_evaluations;
                this->best_y[0] = y;
                this->best_transformed_y[0] = transformed_y;
            }


            void set_parameters(
                const std::vector<std::shared_ptr<double>> &parameters) {
                if (this->logging_parameters.size() != 0)
                    this->logging_parameters.clear();

                for (size_t i = 0; i != parameters.size(); i++)
                    this->logging_parameters[
                            "parameter" + common::to_string(i + 1)] =
                        parameters[i];
            }

            void set_parameters(
                const std::vector<std::shared_ptr<double>> &parameters,
                const std::vector<std::string> &parameters_name) {
                if (parameters_name.size() != parameters.size())
                    common::log::error(
                        "Parameters and their names are given with different size.");

                if (this->logging_parameters.size() != 0)
                    this->logging_parameters.clear();
                for (size_t i = 0; i != parameters.size(); i++)
                    this->logging_parameters[parameters_name[i]] = parameters[i
                    ];
            }

            // Only for python wrapper.
            // Todo make a subclass for this
            void add_dynamic_attribute(
                const std::vector<std::shared_ptr<double>> &attributes) {
                if (this->attr_per_run_name_value.size() != 0)
                    this->attr_per_run_name_value.clear();
                for (size_t i = 0; i != attributes.size(); i++)
                    this->attr_per_run_name_value[
                            "attr" + common::to_string(i + 1)] =
                        attributes[i];
            }


            void add_dynamic_attribute(
                const std::vector<std::shared_ptr<double>> &attributes,
                const std::vector<std::string> &attributes_name) {
                if (attributes_name.size() != attributes.size())
                    common::log::error(
                        "Attributes and their names are given with different size.");

                if (this->attr_per_run_name_value.size() != 0)
                    this->attr_per_run_name_value.clear();
                for (size_t i = 0; i != attributes.size(); i++)
                    this->attr_per_run_name_value[attributes_name[i]] =
                        attributes[i];
            }

            void set_dynamic_attributes_name(
                const std::vector<std::string> &attributes_name) {
                if (this->attr_per_run_name_value.size() != 0)
                    this->attr_per_run_name_value.clear();
                //default value set as -9999.
                for (size_t i = 0; i != attributes_name.size(); i++)
                    this->attr_per_run_name_value[attributes_name[i]] =
                        std::make_shared<
                            double>(-9999);
            }

            void set_dynamic_attributes_name(
                const std::vector<std::string> &attributes_name,
                const std::vector<double> &initial_attributes) {
                if (attributes_name.size() != initial_attributes.size())
                    common::log::error(
                        "Attributes and their names are given with different size.");

                for (size_t i = 0; i != attributes_name.size(); i++)
                    this->attr_per_run_name_value[attributes_name[i]] = std::make_shared<double>(initial_attributes[i]);
            }

            void set_dynamic_attributes(
                const std::vector<std::string> &attributes_name,
                const std::vector<double> &attributes) {
                if (attributes_name.size() != attributes.size())
                    common::log::error(
                        "Attributes and their names are given with different size.");

                for (size_t i = 0; i != attributes_name.size(); ++i)
                    if (this->attr_per_run_name_value.find(attributes_name[i])
                        != this->attr_per_run_name_value.end())
                        *this->attr_per_run_name_value[attributes_name[i]] = attributes[i];
                    else
                        common::log::error(
                            "Dynamic attributes " + attributes_name[i] +
                            " does not exist");
            }

            void set_parameters_name(const std::vector<std::string> &parameters_name) {
                if (this->logging_parameters.size() != 0)
                    this->logging_parameters.clear();

                //default value set as -9999.
                for (size_t i = 0; i != parameters_name.size(); i++)
                    this->logging_parameters[parameters_name[i]] = std::make_shared<double>(-9999);
            }

            void set_parameters_name(const std::vector<std::string> &parameters_name,
                                     const std::vector<double> &initial_parameters) {
                if (parameters_name.size() != initial_parameters.size())
                    common::log::error("Parameters and their names are given with different size.");

                for (size_t i = 0; i != parameters_name.size(); i++)
                    this->logging_parameters[parameters_name[i]] = std::make_shared<double>(initial_parameters[i]);
            }

            void set_parameters(const std::vector<std::string> &parameters_name,
                                const std::vector<double> &parameters) {
                if (parameters_name.size() != parameters.size())
                    common::log::error(
                        "Parameters and their names are given with different size.");

                for (size_t i = 0; i != parameters_name.size(); ++i)
                    if (this->logging_parameters.find(parameters_name[i]) != this->logging_parameters.end())
                        *this->logging_parameters[parameters_name[i]] = parameters[i];
                    else
                        common::log::error(
                            "Parameter " + parameters_name[i] +
                            " does not exist");
            }
        };
    }
}
