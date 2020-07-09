
.. _program_listing_file_src_Template_IOHprofiler_observer.h:

Program Listing for File IOHprofiler_observer.h
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_observer.h>` (``src/Template/IOHprofiler_observer.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_OBSERVER_H
   #define _IOHPROFILER_OBSERVER_H
   
   #include "IOHprofiler_common.h"
   #include "IOHprofiler_problem.h"
   class IOHprofiler_observer {
   public:
     IOHprofiler_observer() :
       observer_interval(0),
       observer_complete_flag(false),
       observer_update_flag(true),
       observer_time_points({0}),
       evaluations_value1(1),
       time_points_index(0),
       time_points_expi(0),
       observer_time_points_exp_base1(10),
       evaluations_value2(1),
       evaluations_expi(0),
       observer_time_points_exp_base2(10){}
     virtual ~IOHprofiler_observer() {};
   
     IOHprofiler_observer(const IOHprofiler_observer &) = delete;
     IOHprofiler_observer &operator = (const IOHprofiler_observer&) = delete;
     
     void set_complete_flag(bool complete_flag);
   
     bool complete_status() const;
   
     bool complete_trigger() const;
     
     void set_interval(int interval);
   
     bool interval_status() const;
   
     bool interval_trigger(size_t evaluations) const;
   
     void set_update_flag(bool update_flag);
   
     bool update_trigger(double fitness, IOH_optimization_type optimization_type);
   
     bool update_status() const;
     
     void set_time_points(const std::vector<int> & time_points, const int number_of_evaluations, const int time_points_exp_base1 = 10, const int time_points_exp_base2 = 10);
   
     bool time_points_status() const;
   
     bool time_points_trigger(size_t evaluations);
   
     void reset_observer(const IOH_optimization_type optimization_type);
   
     virtual void do_log(const std::vector<double> &log_info) {}
   
     virtual void track_problem(const IOHprofiler_problem<int> & problem) {}
     
     virtual void track_problem(const IOHprofiler_problem<double> & problem) {}
   
   private:
     int observer_interval; 
     bool observer_complete_flag; 
     bool observer_update_flag; 
    
     std::vector<int> observer_time_points; 
     size_t evaluations_value1; 
     size_t time_points_index; 
     int time_points_expi; 
     int observer_time_points_exp_base1;
   
     int observer_number_of_evaluations; 
     size_t evaluations_value2; 
     int evaluations_expi; 
     int observer_time_points_exp_base2;
     
     double current_best_fitness;
   };
   
   #endif // _IOHPROFILER_OBSERVER_H
