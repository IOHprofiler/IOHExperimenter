
.. _program_listing_file_src_Template_IOHprofiler_class_generator.h:

Program Listing for File IOHprofiler_class_generator.h
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_class_generator.h>` (``src/Template/IOHprofiler_class_generator.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _IOHPROFILER_PROBLEM_GENERATOR_H
   #define _IOHPROFILER_PROBLEM_GENERATOR_H
   
   #include "IOHprofiler_common.h"
   
   typedef std::string defaultIDKeyType;
   
   template <class manufacturedObj> class genericGenerator {
     
     
     typedef  std::map<std::string, std::shared_ptr<manufacturedObj> (*)()> FN_registry;
     FN_registry registry;
   
     genericGenerator();
     genericGenerator(const genericGenerator&) = delete; 
     genericGenerator &operator=(const genericGenerator&) = delete;
   
   public:
     static genericGenerator &instance();
     
     void regCreateFn(std::string, std::shared_ptr<manufacturedObj> (*)());
     
     std::shared_ptr<manufacturedObj> create(std::string className) const;
   };
   
   template <class ancestorType  ,
             class manufacturedObj,
             typename classIDKey=defaultIDKeyType>
   class registerInFactory {
   public:
       static std::shared_ptr<ancestorType> createInstance() {
           return std::shared_ptr<ancestorType>(manufacturedObj::createInstance());
       }
       
       registerInFactory(const classIDKey id) {
         genericGenerator<ancestorType>::instance().regCreateFn(id, createInstance);
       }
   };
   
   #include "IOHprofiler_class_generator.hpp"
   
   #endif
