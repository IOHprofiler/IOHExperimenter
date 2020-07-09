
.. _program_listing_file_src_Problems_common_used_functions_wmodels.hpp:

Program Listing for File wmodels.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_Problems_common_used_functions_wmodels.hpp>` (``src/Problems/common_used_functions/wmodels.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef _F_WMODELS_H
   #define _F_WMODELS_H
   
   #include "IOHprofiler_random.h"
   
   
   static std::vector<int> dummy(int number_of_variables, double select_rate, long inseed) {
     std::vector<int> position;
     std::vector<int> random_index;
     std::vector<double> random_numbers;
     int temp;
     int select_num = (int)floor((double)(number_of_variables * select_rate));
     
     position.reserve(number_of_variables);
     for (int i = 0; i != number_of_variables; ++i) {
       position.push_back(i);
     }
   
     IOHprofiler_random::IOHprofiler_uniform_rand((size_t)select_num,inseed,random_numbers);
     for (int i = 0; i < select_num; ++i) {
       random_index.push_back((int)floor(random_numbers[i] * 1e4 / 1e4 * number_of_variables));
     }
     for (int i = 0; i != select_num; ++i) {
       temp = position[i];
       position[i] = position[random_index[i]];
       position[random_index[i]] = temp;
     }
   
     sort(position.begin(),position.begin()+select_num);
   
     random_index.clear();
     for(int i = 0; i != select_num; ++i){
       random_index.push_back(position[i]);
     }
     return random_index;
   }
   
   static std::vector<int> neutrality(const std::vector<int> &variables, int mu) {
     int number_of_variables = variables.size();
     int n = (int)floor((double)number_of_variables/(double)mu);
     std::vector<int> new_variables;
   
     new_variables.reserve(n);
     int i = 0, temp = 0;
     while (i != number_of_variables) {
       temp += variables[i];
       if ((i+1) % mu == 0 && i != 0) {
         if (temp >= mu / 2.0) {
           new_variables.push_back(1);
         } else {
           new_variables.push_back(0);
         }
         temp = 0;
       }
       i++;
     }
   
     return new_variables;
   }
   
   static std::vector<int> epistasis(const std::vector<int> &variables, int v) {
     int h, epistasis_result;
     int number_of_variables = variables.size();
     std::vector<int> new_variables;
     new_variables.reserve(number_of_variables);
     h = 0;
     while (h + v - 1 < number_of_variables) {
       int i = 0;
       while(i < v) {
         epistasis_result = -1;
         for (int j = 0; j < v; ++j) {
           if ((v - j - 1) != ((v - i - 1) - 1) % 4) {
             if (epistasis_result == -1) {
               epistasis_result = variables[h+j];
             } else {
               epistasis_result = (epistasis_result != variables[h+j]);
             }
           }
         }
         new_variables.push_back(epistasis_result);
         ++i;
       }
       h += v;
     }
     if(number_of_variables - h > 0) {
       v = number_of_variables - h;
       int i = 0;
       while (i < v) {
         epistasis_result = -1;
         for (int j = 0; j < v; ++j) {
           if ((v - j - 1) != ((v - i - 1) - 1) % 4) {
             if (epistasis_result == -1) {
               epistasis_result = variables[h+j];
             } else {
               epistasis_result = (epistasis_result != variables[h+j]);
             }
           }
         }
         new_variables.push_back(epistasis_result);
         ++i;
       }
       h += v;
     }
     return new_variables;
   }
   
   static double ruggedness1(double y, int number_of_variables) {
     double ruggedness_y, s;
     s = (double)number_of_variables;
     if(y == s) {
       ruggedness_y = ceil(y/2.0) + 1.0;
     } else if (y < s && number_of_variables % 2 == 0) {
       ruggedness_y = floor(y / 2.0) + 1.0;
     } else if (y < s && number_of_variables % 2 != 0) {
       ruggedness_y = ceil(y / 2.0) + 1.0;
     } else {
       ruggedness_y = y;
       assert(y <= s);
     }
     return ruggedness_y;
   }
   
   static double ruggedness2(double y, int number_of_variables) {
     double ruggedness_y;
     int tempy=(int)(y+0.5);
     if (tempy == number_of_variables) {
       ruggedness_y = y;
     } else if (tempy < number_of_variables && tempy % 2 == 0 && number_of_variables % 2 == 0) {
       ruggedness_y = y + 1.0;
     } else if (tempy < number_of_variables && tempy % 2 == 0 && number_of_variables % 2 != 0) {
       ruggedness_y = (y - 1.0) > 0 ? (y - 1.0) : 0;
     } else if (tempy < number_of_variables && tempy % 2 != 0 && number_of_variables % 2 == 0) {
       ruggedness_y = (y - 1.0) > 0 ? (y - 1.0) : 0;
     } else if (tempy < number_of_variables && tempy % 2 != 0 && number_of_variables % 2 != 0) {
       ruggedness_y = y + 1.0;
     } else {
       ruggedness_y = y;
       assert(tempy <= number_of_variables);
     }
     return ruggedness_y;
   }
   
   static std::vector<double> ruggedness3(int number_of_variables) {
     std::vector<double> ruggedness_fitness(number_of_variables+1,0.0);
     
     for (int j = 1; j <= number_of_variables/5; ++j) {
       for (int k = 0; k < 5; ++k) {
         ruggedness_fitness[number_of_variables-5*j+k] = (double)(number_of_variables - 5 * j + (4-k));
       }
     }
     for (int k = 0; k < number_of_variables - number_of_variables / 5 * 5; ++k) {
       ruggedness_fitness[k] = (double)(number_of_variables - number_of_variables / 5 * 5 - 1 - k);
     }
     ruggedness_fitness[number_of_variables] = (double)number_of_variables;
     return ruggedness_fitness;
   }
   
   
   static void layer_neutrality_compute(const std::vector<int> &xIn, std::vector<int> &xOut, const int mu){
     int thresholdFor1 = (mu >> 1) + (mu & 1);
     int i,j,ones,flush;
     int temp;
     int dim = xIn.size();
     int temp_dim = dim / mu;
     if (xOut.size() != temp_dim) {
       xOut.resize(temp_dim);
     }
     i=0;j=0; ones=0; flush=mu;
     while ((i < dim) && (j<temp_dim)){
       if (xIn[i]==1) {
         ones+=1;
       }
       i+=1;
       if (i >= flush) {
         flush+=mu;
         if (ones >= thresholdFor1) {
           temp=1;
         } else {
           temp=0;
         }
         xOut[j]=temp;
         j+=1;
         ones=0;
       }
     }
   }
   
   static void base_epistasis(const std::vector<int> &xIn, const int start, const int nu, std::vector<int> &xOut) {
     const int end = (start + nu) - 1;
     const int flip = xIn[start];
     int result;
     int skip = start;
     for (int i = end ; i >= start; --i) {
       result = flip;
       for (int j = end; j > start; --j) {
         if (j != skip) {
           result ^= (xIn[j]);
         }
       }
       xOut[i] = result;
       if ((--skip) < start) {
         skip = end;
       }
     }
   }
   
   static void epistasis_compute(const std::vector<int>  &xIn, std::vector<int> &xOut, const int nu) {
       const int length = xIn.size();
       const int end = length - nu;
       int i;
       for (i = 0; i <= end; i += nu) {
         base_epistasis(xIn, i, nu, xOut);
       }
       if (i < length) {
         base_epistasis(xIn, i, length - i, xOut);
       }
     }
   
   static void layer_epistasis_compute(const std::vector<int> &x, std::vector<int> &epistasis_x, const int block_size) {
     epistasis_compute(x,epistasis_x,block_size);
   }
   
   static int max_gamma(int q) {
       return (int)((q * (q - 1))>>1);
   }
   
   static std::vector<int> ruggedness_raw(int gamma, int q){
      int i,j,k,start,max;
      int upper;
      int t;
   /*fprintf(stderr,"gamma : %d\n",gamma);
   fprintf(stderr,"q : %d\n",q);*/
      std::vector<int> r(q+1,0);
      r[0]=0;
      max = max_gamma(q);
       if (gamma <= 0){
         start = 0;
       } else {
         start = q - 1 - (int) (0.5 + sqrt(0.25 + ((max - gamma)<<1)));
       }
   /*
   fprintf(stderr,"part1 : %d\n",((int)(0.25 + ((max - gamma)<<1))));
   fprintf(stderr,"part2 : %f\n",(sqrt((int)(0.25 + ((max - gamma)<<1)))));
   fprintf(stderr,"part3 : %d\n",(int)(0.5 + sqrt((int)(0.25 + ((max - gamma)<<1)))));*/
   /*fprintf(stderr,"start %d\n",start);*/
   
   /*for(i=0;i<q;i++){fprintf(stderr,"%d.",r[i]);}fprintf(stderr,"\n");*/
       k = 0;
       for (j = 1; j <= start; j++) {
         if ((j & 1) != 0){ r[j] = (q - k);}
         else{k=k+1; r[j] = k;}
       }
   /*for(i=0;i<q;i++){fprintf(stderr,"%d.",r[i]);}fprintf(stderr,"\n");*/
       for (; j <= q; j++) {
         k=k+1;
         if((start & 1) != 0){r[j] = (q - k);}
         else{r[j]=k;}
       }
       upper = ((gamma - max) +(((q - start - 1) * (q - start)) >> 1));
       j--;
   /*for(i=0;i<q;i++){fprintf(stderr,"%d.",r[i]);}fprintf(stderr,"\n");*/
   /*fprintf(stderr,"upper %d\n",upper);*/
       for (i = 1; i <= upper; i++) {
         j=j-1;
   
         if(j>0){
         t = r[j];
         r[j] = r[q];
         r[q] = t;}
       }
   
   /*fprintf(stderr,"r\n");
   for(i=0;i<=q;i++){fprintf(stderr,"%d %d\n",i, r[i]);}*/
    std::vector<int> r2(q+1,0);
    for(i=0;i<=q;i++){r2[i]=q-r[q-i];}
   /*fprintf(stderr,"r2\n");
   for(i=0;i<=q;i++){fprintf(stderr,"%d %d\n",i, r2[i]);}*/
       return r2;
     }
   
   static int ruggedness_translate(int gamma, int q) {
       int j, k, max, g, lastUpper;
   
       if (gamma <= 0) {
         return 0;
       }
       g = gamma;
       max = max_gamma(q);
       lastUpper = ((q >> 1) * ((q + 1) >> 1));
       if (g <= lastUpper) {
         j = abs(((q + 2) * 0.5)-sqrt((((q * q) * 0.25) + 1) - g));
   
         k = ((g - ((q + 2) * j)) + (j * j) + q);
         return ((k + 1 + ((((q + 2) * j) - (j * j) - q - 1) << 1))- (j - 1));
       }
   
       j = abs((((q % 2) + 1) * 0.5)
           + sqrt((((1 - (q % 2)) * 0.25) + g) - 1 - lastUpper));
   
       k = g - (((j - (q % 2)) * (j - 1)) + 1 + lastUpper);
   
       return (max - k - ((2 * j * j) - j) - ((q % 2) * ((-2 * j) + 1)));
     }
   
   
   
   static double layer_compute_ruggedness(const double y, size_t dimension,int gamma){
     double result;
     std::vector<int> r=ruggedness_raw(ruggedness_translate(gamma, dimension), dimension);
     result=r[ (int)y ];
   
     /*for(i=0;i<dimension+1;i++){fprintf(stderr,"%d ",r[i]);}fprintf(stderr,"\n");
   
     fprintf(stderr,"dim %d val %f, result %f\n",dimension,y,result);*/
     assert(result <= (double) dimension);
     return result;
   }
   
   #endif
