
.. _program_listing_file_src_Problems_BBOB_f_gallagher21.hpp:

Program Listing for File f_gallagher21.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_BBOB_f_gallagher21.hpp>` (``src/Problems/BBOB/f_gallagher21.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_GALLAGHERTWOONE_H
   #define _F_GALLAGHERTWOONE_H
   
   #include "IOHprofiler_problem.h"
   #include "coco_transformation.hpp"
   
   typedef struct f_gallagher_permutation_t{
     double value;
     size_t index;
   }F_GALLAGHER_PERMUTATION_T;
   
   // static bool compareRperm(f_gallagher_permutation_t a, f_gallagher_permutation_t b) {
   //   double temp = a.value - b.value;
   //   if (temp >= 0)
   //     return true;
   //   else
   //     return false;
   // }
   
   static int f_gallagher_compare_doubles(const void *a, const void *b) {
     double temp = (*(f_gallagher_permutation_t *) a).value - (*(f_gallagher_permutation_t *) b).value;
     if (temp > 0)
       return 1;
     else if (temp < 0)
       return -1;
     else
       return 0;
   }
   
   
   class Gallagher21 : public IOHprofiler_problem<double> {
   public:
     Gallagher21(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       IOHprofiler_set_instance_id(instance_id);
       IOHprofiler_set_problem_id(22);
       IOHprofiler_set_problem_name("Gallagher21");
       IOHprofiler_set_problem_type("bbob");
       IOHprofiler_set_number_of_objectives(1);
       IOHprofiler_set_lowerbound(-5.0);
       IOHprofiler_set_upperbound(5.0);
       IOHprofiler_set_best_variables(0);
       IOHprofiler_set_number_of_variables(dimension);
       IOHprofiler_set_as_minimization();
     }
     
     ~Gallagher21() {}
   
     std::vector<double> xopt;
     const size_t number_of_peaks = 21;
     std::vector<std::vector<double> > rotation;
     std::vector<std::vector<double> > arr_scales;
     std::vector<std::vector<double> > x_local;
     std::vector<double> peak_values;
     
     void prepare_problem() {
       const long rseed = (long) (22 + 10000 * this->IOHprofiler_get_instance_id());
       int n = IOHprofiler_get_number_of_variables();
   
       double fopt;
       size_t i, j, k;
       double maxcondition = 1000.;
       /* maxcondition1 satisfies the old code and the doc but seems wrong in that it is, with very high
       * probability, not the largest condition level!!! */
       double maxcondition1 = 1000.;
       std::vector<double> arrCondition;
       double fitvalues[2] = { 1.1, 9.1 };
       /* Parameters for generating local optima. In the old code, they are different in f21 and f22 */
       double b, c;
       /* Random permutation */
       //std::vector<f_gallagher_permutation_t> rperm;
       f_gallagher_permutation_t *rperm;
   
       std::vector<double> random_numbers;
       arr_scales = std::vector<std::vector<double> > (number_of_peaks);
       for (i = 0; i != number_of_peaks; ++i) {
         arr_scales[i] = std::vector<double>(n);
       }
   
       x_local = std::vector<std::vector<double> > (n);
       for (i = 0; i != n; ++i) {
         x_local[i] = std::vector<double>(number_of_peaks);
       }
   
       /* Allocate temporary storage and space for the rotation matrices */
       std::vector<double >xopt = std::vector<double>(n);
       
       b = 9.8;
       c = 4.9;
   
       bbob2009_compute_rotation(rotation, rseed, n);
   
       /* Initialize all the data of the inner problem */
       bbob2009_unif(random_numbers, number_of_peaks - 1, rseed);
       //rperm = std::vector<f_gallagher_permutation_t> (number_of_peaks - 1);
       rperm = (f_gallagher_permutation_t *)malloc((number_of_peaks-1) * sizeof(f_gallagher_permutation_t));
       for (i = 0; i < number_of_peaks - 1; ++i) {
         rperm[i].value = random_numbers[i];
         rperm[i].index = i;
       }
       //std::sort(rperm.begin(), rperm.end(), compareRperm);
       qsort(rperm, number_of_peaks - 1, sizeof(f_gallagher_permutation_t), f_gallagher_compare_doubles);
   
       /* Random permutation */
       arrCondition = std::vector<double>(number_of_peaks);
       arrCondition[0] = maxcondition1;
       peak_values = std::vector<double>(number_of_peaks);
       peak_values[0] = 10;
       for (i = 1; i < number_of_peaks; ++i) {
         arrCondition[i] = pow(maxcondition, (double) (rperm[i - 1].index) / ((double) (number_of_peaks - 2)));
         peak_values[i] = (double) (i - 1) / (double) (number_of_peaks - 2) * (fitvalues[1] - fitvalues[0])
             + fitvalues[0];
       }
   
       free(rperm);
       rperm = (f_gallagher_permutation_t *)malloc((n) * sizeof(f_gallagher_permutation_t));
      
       //rperm = std::vector<f_gallagher_permutation_t> (n);
       for (i = 0; i < number_of_peaks; ++i) {
         bbob2009_unif(random_numbers, n, rseed + (long) (1000 * i));
         for (j = 0; j < n; ++j) {
           rperm[j].value = random_numbers[j];
           rperm[j].index = j;
         }
         //std::sort(rperm.begin(), rperm.end(), compareRperm);
         qsort(rperm, n, sizeof(f_gallagher_permutation_t), f_gallagher_compare_doubles);
         for (j = 0; j < n; ++j) {
           arr_scales[i][j] = pow(arrCondition[i],                             /* Lambda^alpha_i from the doc */
               ((double) rperm[j].index) / ((double) (n - 1)) - 0.5);
         }
       }
       free(rperm);
       bbob2009_unif(random_numbers, n * number_of_peaks, rseed);
       std::vector<double> best_variables(n);
       for (i = 0; i < n; ++i) {
         xopt[i] = 0.8 * (b * random_numbers[i] - c);
         best_variables[i] = 0.8 * (b * random_numbers[i] - c);
         for (j = 0; j < number_of_peaks; ++j) {
           x_local[i][j] = 0.;
           for (k = 0; k < n; ++k) {
             x_local[i][j] += rotation[i][k] * (b * random_numbers[j * n + k] - c);
           }
           if (j == 0) {
             x_local[i][j] *= 0.8;
           }
         }
       }
       IOHprofiler_set_best_variables(best_variables);
        
       fopt = bbob2009_compute_fopt(22, this->IOHprofiler_get_instance_id());
       Coco_Transformation_Data::fopt = fopt;
     }
   
   
     double internal_evaluate(const std::vector<double> &x) {
       int n = x.size();
   
       size_t i, j; /* Loop over dim */
       std::vector<double> tmx(n);
       double a = 0.1;
       double tmp2, f = 0., f_add, tmp, f_pen = 0., f_true = 0.;
       double fac;
       std::vector<double> result(1);
   
   
       fac = -0.5 / (double) n;
   
       /* Boundary handling */
       for (i = 0; i < n; ++i) {
         tmp = fabs(x[i]) - 5.;
         if (tmp > 0.) {
           f_pen += tmp * tmp;
         }
       }
       f_add = f_pen;
       /* Transformation in search space */
       /* TODO: this should rather be done in f_gallagher */
       
       for (i = 0; i < n; i++) {
         tmx[i] = 0;
         for (j = 0; j < n; ++j) {
           tmx[i] += rotation[i][j] * x[j];
         }
       }
       /* Computation core*/
       for (i = 0; i < number_of_peaks; ++i) {
         tmp2 = 0.;
         for (j = 0; j < n; ++j) {
           tmp = (tmx[j] - x_local[j][i]);
           tmp2 += arr_scales[i][j] * tmp * tmp;
         }
         tmp2 = peak_values[i] * exp(fac * tmp2);
         f = f > tmp2 ? f : tmp2;
       }
   
       f = 10. - f;
       if (f > 0) {
         f_true = log(f) / a;
         f_true = pow(exp(f_true + 0.49 * (sin(f_true) + sin(0.79 * f_true))), a);
       } else if (f < 0) {
         f_true = log(-f) / a;
         f_true = -pow(exp(f_true + 0.49 * (sin(0.55 * f_true) + sin(0.31 * f_true))), a);
       } else {
         f_true = f;
       }
   
       f_true *= f_true;
       f_true += f_add;
       result[0] = f_true;
      
       return result[0];
     }
   
     static Gallagher21 * createInstance(int instance_id = DEFAULT_INSTANCE, int dimension = DEFAULT_DIMENSION) {
       return new Gallagher21(instance_id, dimension);
     }
   };
   
   #endif
