
.. _program_listing_file_src_Template_IOHprofiler_platform.h:

Program Listing for File IOHprofiler_platform.h
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Template_IOHprofiler_platform.h>` (``src/Template/IOHprofiler_platform.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef __IOHPROFILER_PLATFORM
   #define __IOHPROFILER_PLATFORM
   
   #include <stddef.h>
   
   
   #if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
     #include <windows.h>
     #include <io.h>
     static const char *IOHprofiler_path_separator = "\\";
   #elif defined(__gnu_linux__)
     #include <linux/limits.h>
     #include <sys/stat.h>
     #include <sys/types.h>
     #include <unistd.h>
     static const char *IOHprofiler_path_separator = "/";
   #elif defined(__APPLE__)
     #include <sys/stat.h>
     #include <sys/syslimits.h>
     #include <sys/types.h>
     #include <unistd.h>
     static const char *IOHprofiler_path_separator = "/";
   #elif defined(__FreeBSD__)
     #include <limits.h>
     #include <sys/stat.h>
     #include <sys/types.h>
     #include <unistd.h>
     static const char *IOHprofiler_path_separator = "/";
   #elif (defined(__sun) || defined(sun)) && (defined(__SVR4) || defined(__svr4__))
     #include <limits.h>
     #include <sys/stat.h>
     #include <sys/types.h>
     #include <unistd.h>
     static const char *IOHprofiler_path_separator = "/";
   #else
     #include <sys/stat.h>
     #include <unistd.h>
     static const char *IOHprofiler_path_separator = "/";
   #endif
   
   /* Definitions needed for creating and removing directories */
   /* Separately handle the special case of Microsoft Visual Studio 2008 with x86_64-w64-mingw32-gcc */
   #if _MSC_VER
   #include <direct.h>
   #elif defined(__MINGW32__) || defined(__MINGW64__)
   #include <dirent.h>
   #else
   #include <dirent.h>
   
   
   
   #endif
   
   /* Definition of the S_IRWXU constant needed to set file permissions */
   #if defined(HAVE_GFA)
   #define S_IRWXU 0700
   #endif
   
   /* To silence the Visual Studio compiler (C4996 warnings in the python build). */
   #ifdef _MSC_VER
   #pragma warning(disable : 4996)
   #endif
   
   #endif
