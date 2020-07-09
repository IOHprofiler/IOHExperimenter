
.. _program_listing_file_src_Problems_IOHprofiler_all_problems.hpp:

Program Listing for File IOHprofiler_all_problems.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_IOHprofiler_all_problems.hpp>` (``src/Problems/IOHprofiler_all_problems.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _IOHPROFILER_ALL_PROBLEMS_HPP
   #define _IOHPROFILER_ALL_PROBLEMS_HPP
   
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
   #include "f_ising_1D.hpp"
   #include "f_ising_2D.hpp"
   #include "f_ising_triangle.hpp"
   #include "f_MIS.hpp"
   #include "f_N_queens.hpp"
   
   #include "IOHprofiler_class_generator.h"
   
   static registerInFactory<IOHprofiler_problem<int>,OneMax> regOneMax("OneMax");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Dummy1> regOneMax_Dummy1("OneMax_Dummy1");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Dummy2> regOneMax_Dummy2("OneMax_Dummy2");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Epistasis> regOneMax_Epistasis("OneMax_Epistasis");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Neutrality> regOneMax_Neutrality("OneMax_Neutrality");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness1> regOneMax_Ruggedness1("OneMax_Ruggedness1");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness2> regOneMax_Ruggedness2("OneMax_Ruggedness2");
   static registerInFactory<IOHprofiler_problem<int>,OneMax_Ruggedness3> regOneMax_Ruggedness3("OneMax_Ruggedness3");
   
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes> regLeadingOnes("LeadingOnes");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Dummy1> regLeadingOnes_Dummy1("LeadingOnes_Dummy1");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Dummy2> regLeadingOnes_Dummy2("LeadingOnes_Dummy2");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Epistasis> regLeadingOnes_Epistasis("LeadingOnes_Epistasis");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Neutrality> regLeadingOnes_Neutrality("LeadingOnes_Neutrality");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness1> regLeadingOnes_Ruggedness1("LeadingOnes_Ruggedness1");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness2> regLeadingOnes_Ruggedness2("LeadingOnes_Ruggedness2");
   static registerInFactory<IOHprofiler_problem<int>,LeadingOnes_Ruggedness3> regLeadingOnes_Ruggedness3("LeadingOnes_Ruggedness3");
   
   static registerInFactory<IOHprofiler_problem<int>,Linear> regLinear("Linear");
   static registerInFactory<IOHprofiler_problem<int>,MIS> regMIS("MIS");
   static registerInFactory<IOHprofiler_problem<int>,LABS> regLABS("LABS");
   static registerInFactory<IOHprofiler_problem<int>,NQueens> regNQueens("NQueens");
   static registerInFactory<IOHprofiler_problem<int>,Ising_1D> regIsing_1D("Ising_1D");
   static registerInFactory<IOHprofiler_problem<int>,Ising_2D> regIsing_2D("Ising_2D");
   static registerInFactory<IOHprofiler_problem<int>,Ising_Triangle> regIsing_Triangle("Ising_Triangle");
   
   #endif //_IOHPROFILER_ALL_PROBLEMS_HPP
