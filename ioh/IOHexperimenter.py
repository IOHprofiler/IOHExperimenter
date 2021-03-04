from .iohcpp import PBO_Logger, BBOB_Logger
from Utils import get_IOH_problem

from itertools import product
from functools import partial
from multiprocessing import cpu_count

import numpy as np
       
def _run_default(alg, fid, dim, iid, precision, suite, repetitions, observing,
        location, foldername, dat, cdat, idat, tdat_base, tdat_exp,
        parameters, dynamic_attrs, static_attrs):
    '''A helper function for parallellization of the IOHexperimter.
    '''
    name = alg.__class__.__name__
    info = "Run using the IOHexperimenter in python, beta version"
    f = IOH_function(fid, dim, iid, precision, suite)
    if observing:
        logger = IOH_logger(location, foldername, name, info)
        logger.set_tracking_options(dat, cdat, idat, tdat_base, tdat_exp)
        if parameters is not None:
            logger.track_parameters(alg, parameters)
        if dynamic_attrs is not None:
            logger.set_dynamic_attributes(dynamic_attrs)
        if static_attrs is not None:
            logger.set_static_attributes(static_attrs)
        f.add_logger(logger)

    for rep in range(repetitions):
        np.random.seed = rep
        if rep > 0:
            f.reset()
        alg(f)
        print(fid, f.evaluations, f.best_so_far_precision)

    f.clear_logger()
    
def _run_custom(alg, function, fname, dim, suite, repetitions, observing,
        location, foldername, dat, cdat, idat, tdat_base, tdat_exp,
        parameters, dynamic_attrs, static_attrs):
    '''A helper function for parallellization of the IOHexperimter.
    '''
    name = alg.__class__.__name__
    info = "Run using the IOHexperimenter in python, beta version"
    f = custom_IOH_function(function, fname, dim)

    if observing:
        logger = IOH_logger(location, foldername, name, info)
        logger.set_tracking_options(dat, cdat, idat, tdat_base, tdat_exp)
        if parameters is not None:
            logger.track_parameters(alg, parameters)
        if dynamic_attrs is not None:
            logger.set_dynamic_attributes(dynamic_attrs)
        if static_attrs is not None:
            logger.set_static_attributes(static_attrs)
        f.add_logger(logger)

    for rep in range(repetitions):
        np.random.seed = rep
        if rep > 0:
            f.reset()
        alg(f)
        print(f.fid, f.evaluations, f.best_so_far_precision)
    f.clear_logger()
                        
class IOHexperimenter():
    '''An interfact to easily run a set of algorithms on multiple functions from the IOHexperimenter.
    '''
    def __init__(self):
        '''Create an IOHexperimenter object for benchmarking a set of algorithms on multiple functions
        Initialize the functions to use by calling 'initialize_PBO', 'initialize_BBOB' or 'initialize_custom'
        Set up parallellization by calling 'set_parallell'
        '''
        self.location = None
        self.foldername = None
        self.dat = True
        self.cdat = False
        self.idat = 0
        self.tdat_base = [0]
        self.tdat_exp = 0
        self.observing = False
        self.parameters = None
        self.dynamic_attrs = None
        self.static_attrs = None
        self.precision = 0
        self.parallel_settings = None
    
    def initialize_PBO(self, fids, iids, dims, repetitions):
        '''Initialize to a set of functions for the PBO suite
        Parameters
        ----------
        fids:
            List of the function numbers within the selected suite. Alternatively, the names of the functions as used in IOHexperimenter.
        dims:
            List of the dimensions (number of variables) of the problem
        iids:
            List of the instance IDs of the problem
        repetitions: 
            Number of repetitions on each function instance
        '''
        self.fids = fids
        self.iids = iids
        self.dims = dims
        self.repetitions = repetitions
        self.suite = "PBO"
    
    def initialize_BBOB(self, fids, iids, dims, repetitions, precision = 1e-8):
        '''Initialize to a set of functions for the BBOB suite
        Parameters
        ----------
        fids:
            List of the function numbers within the selected suite. Alternatively, the names of the functions as used in IOHexperimenter.
        dims:
            List of the dimensions (number of variables) of the problem
        iids:
            List of the instance IDs of the problem
        repetitions: 
            Number of repetitions on each function instance
        target_precision:
            Optional, how close to the optimum the problem is considered 'solved'
        '''
        self.fids = fids
        self.iids = iids
        self.dims = dims
        self.repetitions = repetitions
        self.precision = precision
        self.suite = "BBOB"
    
    def initialize_custom(self, functions, fnames, fdims, repetitions, suite = "No Suite"):
        '''Initialize to a set of custom functions
        Parameters
        ----------
        functions:
            List of functions to use
        fnames:
            List of names corresponding to the provided functions
        fdims:
            List of dimensionalities corresponding to the provided functions
        repetitions: 
            Number of repetitions on each function instance
        suite:
            Optional, a name for the suite of the provided functions
        '''
        self.functions = functions
        self.fnames = fnames
        self.fdims = fdims
        self.repetitions = repetitions
        self.suite = suite
    
    def set_logger_location(self, location, foldername = "run"):
        '''Set the location options for the logger of the suite
        Parameters
        ----------
        location:
            The directory in which to store the results
        foldername:
            The name of the folder to create to store the results
        
        '''
        self.location = location
        self.foldername = foldername
        self.observing = True
        
    def set_parallel(self, parallel, version = "joblib", timeout = 30, num_threads = None):
        '''Set the parallellization options for the experiments
        
        Parameters
        ----------
        parallel:
            Boolean. Whether or not to use parallell execution
        version:
            Which parallellization library to use. Options are:
                - 'multiprocessing'
                - 'MPI' (using the schwimmbad library)
                - 'joblib' (recommended) 
                - 'pebble' (can timeout functions which take too long)
        num_threads:
            How many threads to use. Defaults to all available ones from 'cpu_count'
        timeout:
            If using pebble, this sets the timeout in seconds after which to cancel the execution
        '''
        self.parallel_settings = {
            "evaluate_parallel" : parallel, "use_MPI" : version == "MPI", 
            "use_pebble" : version == "pebble", "timeout" : timeout, 
            "use_joblib" : version == "joblib",
            "num_threads" : num_threads if num_threads is not None else cpu_count()
        }
        
    def set_logger_options(self, dat = True, cdat = False, idat = 0, tdat_base = [0], tdat_exp = 0):
        '''Set which datafiles should be stored by the logger
        Parameters
        ----------
        dat:
            Whether or not to store .dat-files (one line per improvement of objective)
        cdat:
            Whether or not to store .cdat-files (one line for each evaluation of objective)
        idat:
            Integer, if non-zero, an .idat file will be made which records ever i-th evaluation of the objective
        tdat_base:
            Base values for the tdat-file
        tdat_exp:
            Exponent-values for the tdat-file
        '''
        self.dat = dat
        self.cdat = cdat
        self.idat = idat
        self.tdat_base = tdat_base
        self.tdat_exp = tdat_exp
        
    def set_parameter_tracking(self, parameters):
        '''Set which parameters should be tracked during the benchmarking procedure
        
        Only usable when 'set_logger_location' has been called previously
        
        Parameters
        ----------
        parameters:
            List of the names of the parameters to track. All 'parameters'-variables need to be accessible by 
            using algorithm.parameter (recommended to use @property) for each of the algorithms to be benchmarked
        '''
        self.parameters = parameters
    
    def set_dynamic_tracking(self, attributes):
        '''Set which attributes should be tracked for each algorithm run (stored at the end of the run in the meta-information)
        
        Only usable when 'set_logger_location' has been called previously
        
        Parameters
        ----------
        attributes:
            List of the names of the parameters to track. All 'attributes'-variables need to be accessible by 
            using algorithm.parameter (recommended to use @property) for each of the algorithms to be benchmarked
        '''        
        self.dynamic_attrs = attributes

    def set_static_tracking(self, attributes):
        '''Set which static attributes should be stored in the meta-information of the benchmark data
        
        Only usable when 'set_logger_location' has been called previously
        
        Parameters
        ----------
        attributes:
            List of the names of the parameters to track. All 'attributes'-variables need to be accessible by 
            using algorithm.parameter (recommended to use @property) for each of the algorithms to be benchmarked
        '''        
        self.static_attrs = attributes
    
    def __call__(self, algorithms):
        '''Run the experiment with the given set of algorithms
        Parameters
        ----------
        algorithms:
            A list containing algorithm functions. 
            These will get as input only one argument, of type IOH_function.
        '''
        if self.suite in ["PBO", "BBOB"]:
            arguments = product(algorithms, self.fids, self.dims, self.iids)
            partial_run = partial(_run_default, precision = self.precision, suite = self.suite, 
                      repetitions = self.repetitions, observing = self.observing, 
                      location = self.location, foldername = self.foldername,
                      dat = self.dat, cdat = self.cdat, 
                      idat = self.idat, tdat_base = self.tdat_base, 
                      tdat_exp = self.tdat_exp, parameters = self.parameters,
                      dynamic_attrs = self.dynamic_attrs, static_attrs = self.static_attrs)
        else:
            arguments = product(algorithms, self.functions, self.fnames, self.fdims)
            partial_run = partial(_run_custom, suite = self.suite, 
                      repetitions = self.repetitions, observing = self.observing, 
                      location = self.location, foldername = self.foldername,
                      dat = self.dat, cdat = self.cdat, 
                      idat = self.idat, tdat_base = self.tdat_base, 
                      tdat_exp = self.tdat_exp, parameters = self.parameters,
                      dynamic_attrs = self.dynamic_attrs, static_attrs = self.static_attrs)
            
        results = runParallelFunction(partial_run, arguments, self.parallel_settings)   
        return results