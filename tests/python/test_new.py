import unittest
import ioh

class Add2Test(unittest.TestCase):
    def test_add(self):
        s = ioh.Sphere(1, 5)
        self.assertGreater(s.evaluate([0,0,0,0,0]), 0)
        self.assertEqual(3, ioh.add(1, 2))

    def test_to_lower(self):
        self.assertEqual("hallo", ioh.to_lower("HALLO"))

    # def init_sphere(self):
    #     s = ioh.Shpere(1, 5)
    #     self.assertGreater(s([0,0,0,0,0]), 0)

    def test_all_bbob(self):
        bbob_fid_dict = {
            1: ioh.Sphere, 2: ioh.Ellipsoid, 3: ioh.Rastrigin, 4: ioh.Bueche_Rastrigin, 
            5: ioh.Linear_Slope, 6: ioh.AttractiveSector, 7: ioh.Step_Ellipsoid, 
            8: ioh.Rosenbrock, 9: ioh.Rosenbrock_Rotated, 10: ioh.Ellipsoid_Rotated,
            11: ioh.Discus, 12: ioh.Bent_Cigar, 13: ioh.Sharp_Ridge, 14: ioh.Different_Powers,
            15: ioh.Rastrigin_Rotated, 16: ioh.Weierstrass, 17: ioh.Schaffers10, 
            18: ioh.Schaffers1000, 19: ioh.Griewank_RosenBrock, 20: ioh.Schwefel,
            21: ioh.Gallagher101, 22: ioh.Gallagher21, 23: ioh.Katsuura, 24: ioh.Lunacek_Bi_Rastrigin
        }
        for fct in bbob_fid_dict.values():
            f = fct(1,5)
            self.assertGreater(f.evaluate([0,0,0,0,0]), -1000)

    def test_all_bbob(self):
        pbo_fid_dict = {
            1: ioh.OneMax, 2: ioh.LeadingOnes, 3: ioh.Linear, 4: ioh.OneMax_Dummy1, 
            5: ioh.OneMax_Dummy2, 6: ioh.OneMax_Neutrality, 7: ioh.OneMax_Epistasis, 
            8: ioh.OneMax_Ruggedness1, 9: ioh.OneMax_Ruggedness2, 10: ioh.OneMax_Ruggedness3,
            11: ioh.LeadingOnes_Dummy1, 12: ioh.LeadingOnes_Dummy2, 13: ioh.LeadingOnes_Neutrality,
            14: ioh.LeadingOnes_Epistasis, 15: ioh.LeadingOnes_Ruggedness1, 
            16: ioh.LeadingOnes_Ruggedness2, 17: ioh.LeadingOnes_Ruggedness3, 18: ioh.LABS, 
            19: ioh.MIS, 20: ioh.Ising_Ring, 21: ioh.Ising_Torus, 22: ioh.Ising_Triangular,
            23: ioh.NQueens
        }
        for fct in pbo_fid_dict.values():
            f = fct(1,4)
            self.assertGreater(f.evaluate([0,0,0,0]), -1000)    

    def test_initiate_logger(self):
        log = ioh.PBO_Logger("temp/")
        f = ioh.OneMax(1,16)
        log.track_problem(f)
        f.evaluate(16*[0])
        log.do_log(f.logger_info())
        f.evaluate(16*[1])
        log.do_log(f.logger_info())
        log2 = ioh.BBOB_Logger("temp2/")

    def test_custom_func(self):
        class temp_func(ioh.Custom_IOH_Function):
            def __init__(self):
                ioh.Custom_IOH_Function.__init__(self)
                self.number_of_variables = 5
                self.upperbound = 5*[5]
                self.lowerbound = 5*[-5]
                self.name = "tempfunc"
                self.iid = 1
                self.fid = 1

            # def prepare(self):
                
            def internal_evaluate(self, x):
                return sum(x)

        f = temp_func()
        f.reset()
        self.assertEqual(f.internal_evaluate([1,3,5,5,5]),19)
        self.assertEqual(f.number_of_variables, 5)
        self.assertEqual(f.evaluate([1,3,5,5,5]),19)
        

if __name__ == "__main__":
    unittest.main()