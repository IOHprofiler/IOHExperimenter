
.. _program_listing_file_src_Template_IOHprofiler_class_generator.hpp:

Program Listing for File IOHprofiler_class_generator.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_class_generator.hpp>` (``src/Template/IOHprofiler_class_generator.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   template <class manufacturedObj> genericGenerator<manufacturedObj>::genericGenerator() {
   }
   
   template <class manufacturedObj> genericGenerator<manufacturedObj> &genericGenerator<manufacturedObj>::instance() {
       static genericGenerator theInstance;
       return theInstance;
   }
   
   template <class manufacturedObj> void genericGenerator<manufacturedObj>::regCreateFn(std::string clName, std::shared_ptr<manufacturedObj> (*func)()) {
     registry[clName]=func;
   }
   
   template <class manufacturedObj> std::shared_ptr<manufacturedObj> genericGenerator<manufacturedObj>::create(std::string className) const {
     std::shared_ptr<manufacturedObj> ret(nullptr);
   
     typename FN_registry::const_iterator regEntry = registry.find(className);
     if (regEntry != registry.end()) {
       return (*regEntry).second();
     }
     return ret;
   }
