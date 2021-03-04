from functools import partial
from multiprocessing import cpu_count
from .iohcpp import ioh
import numpy as np

_pbo_fid_dict = {
    1: ioh.OneMax, 2: ioh.LeadingOnes, 3: ioh.Linear, 4: ioh.OneMax_Dummy1, 
    5: ioh.OneMax_Dummy2, 6: ioh.OneMax_Neutrality, 7: ioh.OneMax_Epistasis, 
    8: ioh.OneMax_Ruggedness1, 9: ioh.OneMax_Ruggedness2, 10: ioh.OneMax_Ruggedness3,
    11: ioh.LeadingOnes_Dummy1, 12: ioh.LeadingOnes_Dummy2, 13: ioh.LeadingOnes_Neutrality,
    14: ioh.LeadingOnes_Epistasis, 15: ioh.LeadingOnes_Ruggedness1, 
    16: ioh.LeadingOnes_Ruggedness2, 17: ioh.LeadingOnes_Ruggedness3, 18: ioh.LABS, 
    19: ioh.MIS, 20: ioh.Ising_Ring, 21: ioh.Ising_Torus, 22: ioh.Ising_Triangular,
    23: ioh.NQueens
}

_bbob_fid_dict = {
    1: ioh.Sphere, 2: ioh.Ellipsoid, 3: ioh.Rastrigin, 4: ioh.Bueche_Rastrigin, 
    5: ioh.Linear_Slope, 6: ioh.AttractiveSector, 7: ioh.Step_Ellipsoid, 
    8: ioh.Rosenbrock, 9: ioh.Rosenbrock_Rotated, 10: ioh.Ellipsoid_Rotated,
    11: ioh.Discus, 12: ioh.Bent_Cigar, 13: ioh.Sharp_Ridge, 14: ioh.Different_Powers,
    15: ioh.Rastrigin_Rotated, 16: ioh.Weierstrass, 17: ioh.Schaffers10, 
    18: ioh.Schaffers1000, 19: ioh.Griewank_RosenBrock, 20: ioh.Schwefel,
    21: ioh.Gallagher101, 22: ioh.Gallagher21, 23: ioh.Katsuura, 24: ioh.Lunacek_Bi_Rastrigin
}

def get_function(fid, dim, iid, suite):
    if isinstance(fid, int):
        if suite == "BBOB":
            return _bbob_fid_dict[fid](iid, dim)
        elif suite == "PBO":
            if fid in [21, 23]:
                if not np.sqrt(dim).is_integer():
                    raise Exception("For this function, the dimension needs to be a perfect square!")
            return _pbo_fid_dict[fid](iid, dim)
        else:
            raise Exception("This suite is not yet supported")
    else:
        if fid in ["Ising_2D", "NQueens"]:
            if not np.sqrt(dim).is_integer():
                raise Exception("For this function, the dimension needs to be a perfect square!")
        exec (f"return ioh.{fid}({iid}, {dim})")



def runParallelFunction(runFunction, arguments, parallel_config = None):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system
        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :param force_single:Force the run to be single-threaded, ignoring all settings from IOH_config
        :param timeout:     How many seconds each execution get before being stopped. 
                            Only used when running in parallel without mpi.
        :return:
    """
    if parallel_config is None:
        return runSingleThreaded(runFunction, arguments)    

    if parallel_config['evaluate_parallel']:
        if parallel_config['use_MPI']:
            return runMPI(runFunction, arguments)
        elif parallel_config['use_pebble']:
            return runPebblePool(
                runFunction, arguments, 
                timeout = parallel_config['timeout'], 
                num_threads = parallel_config['num_threads']
            )
        elif parallel_config['use_joblib']:
            return runJoblib(
                runFunction, arguments, 
                num_threads = parallel_config['num_threads']
            )
        else:
            return runPool(
                runFunction, arguments, 
                num_threads = parallel_config['num_threads']
            )
    else:
        return runSingleThreaded(runFunction, arguments)

# Inline function definition to allow the passing of multiple arguments to 'runFunction' through 'Pool.map'
def func_star(a_b, func):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)

def runPool(runFunction, arguments, num_threads = None):
    """
        Small overhead-function to handle multi-processing using Python's built-in multiprocessing.Pool
        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    if num_threads is None:
        num_threads = cpu_count()
    arguments = list(arguments)
    print(f"Running pool with {min(num_threads, len(arguments))} threads")
    from multiprocessing import Pool
    p = Pool(min(num_threads, len(arguments)))

    local_func = partial(func_star, func=runFunction)
    results = p.map(local_func, arguments)
    p.close()
    return results

def runPebblePool(runfunction, arguments, timeout = 30, num_threads = None):
    from pebble import ProcessPool
    from concurrent.futures import TimeoutError
    
    if num_threads is None:
        num_threads = cpu_count()
    
    arguments = list(arguments)
    
    print(f"Running pebble pool with {min(num_threads, len(arguments))} threads")

    
    def task_done(future):
        try:
            future.result()  # blocks until results are ready
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised %s" % error)
            print(error)  # traceback of the function
            
    with ProcessPool(max_workers = min(num_threads, len(arguments)), max_tasks = len(arguments)) as pool:
        for x in arguments:
            if type(x) is not list and type(x) is not tuple:
                x = [x]
            future = pool.schedule(runfunction, args=x, timeout=timeout)
            future.add_done_callback(task_done)

def runJoblib(runFunction, arguments, num_threads = None):
    from joblib import delayed, Parallel
    
    if num_threads is None:
        num_threads = cpu_count()
#     arguments = list(arguments)
#     print(f"Running joblib with {min(num_threads, len(arguments))} threads")
    local_func = partial(func_star, func=runFunction)

    return Parallel(n_jobs = num_threads)(delayed(local_func)(arg) for arg in arguments)

def runSingleThreaded(runFunction, arguments):
    """
        Small overhead-function to iteratively run a function with a pre-determined input arguments
        :param runFunction: The (``partial``) function to run, accepting ``arguments``
        :param arguments:   The arguments to passed to ``runFunction``, one run at a time
        :return:            List of any results produced by ``runFunction``
    """
    results = []
    print("Running single-threaded")
    for arg in arguments:
        results.append(runFunction(*arg))
    return results

def runMPI(runFunction, arguments):
    from schwimmbad import MPIPool
    print("Running using MPI")
    with MPIPool() as pool:
        results = pool.map(runFunction, arguments)
    return results