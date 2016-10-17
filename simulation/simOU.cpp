#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h++>
#include <SDESolvers.hpp>
#include <ODESolvers.hpp>

using namespace libconfig;


/** \file simOU.cpp 
 *  \ingroup examples
 *  \brief Simulate a multi-dimensional Ornstein-Uhlenbeck process.
 *
 *  Simulate a multi-dimensional Ornstein-Uhlenbeck process.
 */


/*
 * Declarations
 */

/** \brief User defined function to get parameters from a cfg file using libconfig. */
void readConfig(const char *cfgFileName);

// Configuration
char caseName[256];       //!< Name of the case to simulate 
char fileFormat[256];    //!< File format of output ("txt" or "bin")
char resDir[256];         //!< Root directory in which results are written
int dim;                  //!< Dimension of the phase space
gsl_matrix *A;            //!< Matrix of the linear drift
gsl_matrix *Q;            //!< Diffusion matrix
gsl_vector *initState;    //!< Initial state
double LCut;              //!< Length of the time series without spinup
double spinup;            //!< Length of initial spinup period to remove
double L;                 //!< Total length of integration
double dt;                //!< Time step of integration
double printStep;         //!< Time step of output
size_t printStepNum;      //!< Time step of output in number of time steps of integration
char dstFileName[256];    //!< Destination file name


/** \brief Simulation of an Ornstein-Uhlenbeck process.
 *
 *  Simulation of an Ornstein-Uhlenbeck process.
 *  After parsing the configuration file,
 *  a linear vector field for the drift, a diffusion matrix
 *  and an Euler-Maruyama stochastic numerical scheme are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  FILE *dstStream;
  gsl_matrix *X;

  // Read configuration file
  try
    {
      readConfig(argv[1]);
    }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  // Open destination ile
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }

  // Configure random number generator
  gsl_rng_env_setup(); // Get GSL_RNG_TYPE [=gsl_rng_mt19937] and GSL_RNG_SEED [=0]
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
  std::cout << "---Random number generator---" << std::endl;
  std::cout << "Generator type: " << gsl_rng_name (r) << std::endl;
  std::cout << "Seed = " <<  gsl_rng_default_seed << std::endl;
  std::cout << std::endl;

  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new linearField(A);

  // Define stochastic vector field
  vectorFieldStochastic *stocField = new additiveWiener(Q, r);

  // Define numerical scheme
  std::cout << "Defining stochastic numerical scheme..." << std::endl;
  numericalSchemeStochastic *scheme = new EulerMaruyama(dim, dt);

  // Define model
  std::cout << "Defining stochastic model..." << std::endl;
  modelStochastic *mod = new modelStochastic(field, stocField, scheme, initState);
  
  // Numerical integration
  printf("Integrating simulation...\n");
  X = mod->integrateForward(L, spinup, printStepNum);

  // Write results
  printf("Writing...\n");
  if (strcmp(fileFormat, "bin") == 0)
    gsl_matrix_fwrite(dstStream, X);
  else
    gsl_matrix_fprintf(dstStream, X, "%f");
  fclose(dstStream);  

  // Free
  gsl_matrix_free(X);
  gsl_vector_free(initState);
  gsl_matrix_free(A);
  gsl_matrix_free(Q);

  return 0;
}


/**
 * Sparse configuration file using libconfig++
 * to define all parameters of the case.
 */
void
readConfig(const char *cfgFileName)
{
  Config cfg;
  
  // Read the file. If there is an error, report it and exit.
  try {
    std::cout << "Reading config file " << cfgFileName << std::endl;
    cfg.readFile(cfgFileName);

    std::cout.precision(6);
    std::cout << "Settings:" << std::endl;
    
      /** Get paths */
    std::cout << std::endl << "---general---" << std::endl;
    strcpy(resDir, (const char *) cfg.lookup("general.resDir"));
    std::cout << "Results directory: " << resDir << std::endl;
        
    /** Get model settings */
    std::cout << std::endl << "---model---" << std::endl;

    // Case name
    strcpy(caseName, (const char *) cfg.lookup("model.caseName"));
    std::cout << "Case name: " << caseName << std::endl;
    
    // Dimension
    dim = cfg.lookup("model.dim");
    std::cout << "dim = " << dim << std::endl;
    
    // Get name of linear drift of OU and its read matrix from file
    A = gsl_matrix_alloc(dim, dim);
    const Setting &driftSetting = cfg.lookup("model.drift");
    std::cout << "Linear drift matrix A = [";
    for (size_t i = 0; i < (size_t) dim; i++)
      {
	std::cout << "[";
	for (size_t j = 0; j < (size_t) dim; j++)
	  {
	    gsl_matrix_set(A, i, j, driftSetting[j + i * dim]);
	    std::cout << gsl_matrix_get(A, i, j) << " ";
	  }
	std::cout << "]";
      }
    std::cout << "]" << std::endl;

    // Get name of diffusion and its read matrix from file
    Q = gsl_matrix_alloc(dim, dim);
    const Setting &diffusionSetting = cfg.lookup("model.diffusion");
    std::cout << "Diffusion matrix Q = [";
    for (size_t i = 0; i < (size_t) dim; i++)
      {
	std::cout << "[";
	for (size_t j = 0; j < (size_t) dim; j++)
	  {
	    gsl_matrix_set(Q, i, j, diffusionSetting[j + i * dim]);
	    std::cout << gsl_matrix_get(Q, i, j) << " ";
	  }
	std::cout << "]";
      }
    std::cout << "]" << std::endl;

    /** Get simulation settings */
    std::cout << "\n" << "---simulation---" << std::endl;

    // Initial state
    const Setting &initStateSetting = cfg.lookup("simulation.initState");
    initState = gsl_vector_alloc(dim);
    std::cout << "initState = [";
    for (size_t i =0; i < (size_t) dim; i++)
      {
	gsl_vector_set(initState, i, initStateSetting[i]);
	std::cout << gsl_vector_get(initState, i) << " ";
      }
    std::cout << "]" << std::endl;

    // Simulation length without spinup
    LCut = cfg.lookup("simulation.LCut");
    std::cout << "LCut = " << LCut << std::endl;

    // Time step
    dt = cfg.lookup("simulation.dt");
    std::cout << "dt = " << dt << std::endl;

    // Spinup period to remove
    spinup = cfg.lookup("simulation.spinup");
    std::cout << "spinup = " << spinup << std::endl;

    // Sub-printStep 
    printStep = cfg.lookup("simulation.printStep");
    std::cout << "printStep = " << printStep << std::endl;

    // Output format
    strcpy(fileFormat, (const char *) cfg.lookup("simulation.fileFormat"));
    std::cout << "Output file format: " << fileFormat << std::endl;

  std::cout << std::endl;

    // Some parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);

    // Define names and open destination file
    sprintf(dstFileName, "%s/simulation/sim%s.%s", resDir, postfix, fileFormat);

  }
  catch(const FileIOException &fioex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw fioex;
  }
  catch(const ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    throw pex;
  }
  catch(const SettingNotFoundException &nfex) {
    std::cerr << "Setting " << nfex.getPath() << " not found." << std::endl;
    throw nfex;
  }
  catch(const SettingTypeException &stex) {
    std::cerr << "Setting type exception." << std::endl;
    throw stex;
  }

  return;
}

