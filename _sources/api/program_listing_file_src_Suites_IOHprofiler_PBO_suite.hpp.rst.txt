
.. _program_listing_file_src_Suites_IOHprofiler_PBO_suite.hpp:

Program Listing for File IOHprofiler_PBO_suite.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Suites_IOHprofiler_PBO_suite.hpp>` (``src/Suites/IOHprofiler_PBO_suite.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_PBO_SUITE_HPP
   #define _IOHPROFILER_PBO_SUITE_HPP
   
   #include "f_one_max.hpp"
   #include "f_leading_ones.hpp"
   #include "f_linear.hpp"
   #include "f_one_max_dummy1.hpp"
   #include "f_one_max_dummy2.hpp"
   #include "f_one_max_neutrality.hpp"
   #include "f_one_max_epistasis.hpp"
   #include "f_one_max_ruggedness1.hpp"
   #include "f_one_max_ruggedness2.hpp"
   #include "f_one_max_ruggedness3.hpp"
   #include "f_leading_ones_dummy1.hpp"
   #include "f_leading_ones_dummy2.hpp"
   #include "f_leading_ones_neutrality.hpp"
   #include "f_leading_ones_epistasis.hpp"
   #include "f_leading_ones_ruggedness1.hpp"
   #include "f_leading_ones_ruggedness2.hpp"
   #include "f_leading_ones_ruggedness3.hpp"
   #include "f_labs.hpp"
   #include "f_ising_ring.hpp"
   #include "f_ising_torus.hpp"
   #include "f_ising_triangular.hpp"
   #include "f_MIS.hpp"
   #include "f_N_queens.hpp"
   
   #include "IOHprofiler_suite.h"
   
   class PBO_suite : public IOHprofiler_suite<int> {
   public:
     PBO_suite() {
       std::vector<int> problem_id;
       std::vector<int> instance_id;
       std::vector<int> dimension;
       for (int i = 0; i < 23; ++i) {
         problem_id.push_back(i+1);
       }
       for (int i = 0; i < 1; ++i) {
         instance_id.push_back(i+1);
       }
       dimension.push_back(100);
       
       IOHprofiler_set_suite_problem_id(problem_id);
       IOHprofiler_set_suite_instance_id(instance_id);
       IOHprofiler_set_suite_dimension(dimension);
       IOHprofiler_set_suite_name("PBO");
       registerProblem();
     }
   
     PBO_suite(std::vector<int> problem_id, std::vector<int> instance_id, std::vector<int> dimension) {
       for (int i = 0; i < problem_id.size(); ++i) {
         if (problem_id[i] < 0 || problem_id[i] > 23) {
           IOH_error("problem_id " + std::to_string(problem_id[i]) + " is not in PBO_suite");
         }
       }
       
       for (int i = 0; i < instance_id.size(); ++i) {
         if (instance_id[i] < 0 || instance_id[i] > 100) {
           IOH_error("instance_id " + std::to_string(instance_id[i]) + " is not in PBO_suite");
         }
       }
   
       for (int i = 0; i < dimension.size(); ++i) {
         if (dimension[i] < 0 || dimension[i] > 20000) {
           IOH_error("dimension " + std::to_string(dimension[i]) + " is not in PBO_suite");
         }
       }
   
       IOHprofiler_set_suite_problem_id(problem_id);
       IOHprofiler_set_suite_instance_id(instance_id);
       IOHprofiler_set_suite_dimension(dimension);
       IOHprofiler_set_suite_name("PBO");
       registerProblem();
     }
   
     void registerProblem() {
       registerInFactory<IOHprofiler_problem<int>,OneMax> regOneMax("OneMax");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Dummy1> regOneMax_Dummy1("OneMax_Dummy1");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Dummy2> regOneMax_Dummy2("OneMax_Dummy2");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Epistasis> regOneMax_Epistasis("OneMax_Epistasis");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Neutrality> regOneMax_Neutrality("OneMax_Neutrality");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness1> regOneMax_Ruggedness1("OneMax_Ruggedness1");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness2> regOneMax_Ruggedness2("OneMax_Ruggedness2");
       registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness3> regOneMax_Ruggedness3("OneMax_Ruggedness3");
   
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes> regLeadingOnes("LeadingOnes");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Dummy1> regLeadingOnes_Dummy1("LeadingOnes_Dummy1");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Dummy2> regLeadingOnes_Dummy2("LeadingOnes_Dummy2");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Epistasis> regLeadingOnes_Epistasis("LeadingOnes_Epistasis");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Neutrality> regLeadingOnes_Neutrality("LeadingOnes_Neutrality");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness1> regLeadingOnes_Ruggedness1("LeadingOnes_Ruggedness1");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness2> regLeadingOnes_Ruggedness2("LeadingOnes_Ruggedness2");
       registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness3> regLeadingOnes_Ruggedness3("LeadingOnes_Ruggedness3");
       
       registerInFactory<IOHprofiler_problem<int>,Linear> regLinear("Linear");
       registerInFactory<IOHprofiler_problem<int>,MIS> regMIS("MIS");
       registerInFactory<IOHprofiler_problem<int>,LABS> regLABS("LABS");
       registerInFactory<IOHprofiler_problem<int>,NQueens> regNQueens("NQueens");
       registerInFactory<IOHprofiler_problem<int>,Ising_Ring> regIsing_Ring("Ising_Ring");
       registerInFactory<IOHprofiler_problem<int>,Ising_Torus> regIsing_Torus("Ising_Torus");
       registerInFactory<IOHprofiler_problem<int>,Ising_Triangular> regIsing_Triangular("Ising_Triangular");
     
       mapIDTOName(1,"OneMax");
       mapIDTOName(2,"LeadingOnes");
       mapIDTOName(3,"Linear");
       mapIDTOName(4,"OneMax_Dummy1");
       mapIDTOName(5,"OneMax_Dummy2");
       mapIDTOName(6,"OneMax_Neutrality");
       mapIDTOName(7,"OneMax_Epistasis");
       mapIDTOName(8,"OneMax_Ruggedness1");
       mapIDTOName(9,"OneMax_Ruggedness2");
       mapIDTOName(10,"OneMax_Ruggedness3");
       mapIDTOName(11,"LeadingOnes_Dummy1");
       mapIDTOName(12,"LeadingOnes_Dummy2");
       mapIDTOName(13,"LeadingOnes_Neutrality");
       mapIDTOName(14,"LeadingOnes_Epistasis");
       mapIDTOName(15,"LeadingOnes_Ruggedness1");
       mapIDTOName(16,"LeadingOnes_Ruggedness2");
       mapIDTOName(17,"LeadingOnes_Ruggedness3");
       mapIDTOName(18,"LABS");
       mapIDTOName(22,"MIS");
       mapIDTOName(19,"Ising_Ring");
       mapIDTOName(20,"Ising_Torus");
       mapIDTOName(21,"Ising_Triangular");
       mapIDTOName(23,"NQueens");
     }
   
     static PBO_suite * createInstance() {
       return new PBO_suite();
     }
   
     static PBO_suite * createInstance(std::vector<int> problem_id, std::vector<int> instance_id, std::vector<int> dimension) {
       return new PBO_suite(problem_id, instance_id, dimension);
     }
   };
   #endif //_IOHPROFILER_PBO_SUITE_H
