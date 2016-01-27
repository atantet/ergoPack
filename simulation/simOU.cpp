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
#include <ergoPack/SDESolvers.hpp>
#include <ergoPack/ODESolvers.hpp>

using namespace libconfig;


/** \file simOU.cpp 
 *  \brief Simulate an multi-dimensional Ornstein-Uhlenbeck process.
 *
 *  Simulate an multi-dimensional Ornstein-Uhlenbeck process.
 */

// Declarations
/** \brief User defined function to get parameters from a cfg file using libconfig. */
int readConfig(const char *cfgFileNamePrefix);

// Paths
const char prefix[] = "";
const char cfgDir[] = "../cfg/";
const char simDir[] = "../results/simulations/";

// Configuration
char caseName[256], file_format[256];
Config cfg;
int dim;
gsl_matrix *A, *Q;
gsl_vector *initState;
double LCut, dt, spinup;
double printStep;
size_t printStepNum;


// Main program
int main(int argc, char * argv[])
{
  double L;
  char postfix[256], dstFileName[256];
  FILE *dstStream;
  gsl_matrix *X;

  // Read configuration file given as first command-line argument
  if (readConfig(argv[1])) {
    std::cerr << "Error reading config file " << argv[1] << ".cfg"
	      << std::endl;
    return(EXIT_FAILURE);
  }

  // Some parameters
  L = LCut + spinup;
  printStepNum = (size_t) (printStep / dt);

  // Define names and open destination file
  sprintf(postfix, "_%s_L%d_spinup%d_dt%d_samp%d", caseName,
	  (int) L, (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)),
	  (int) printStepNum);
  sprintf(dstFileName, "%s/sim%s.%s", simDir, postfix, file_format);
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
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalSchemeStochastic *scheme = new EulerMaruyama(dim, dt);

  // Define model
  std::cout << "Defining model..." << std::endl;
  modelStochastic *mod = new modelStochastic(field, stocField, scheme, initState);
  
  // Numerical integration
  printf("Integrating simulation...\n");
  X = mod->integrateForward(L, spinup, printStepNum);

  // Write results
  printf("Writing...\n");
  if (strcmp(file_format, "bin") == 0)
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


// Definitions
int
readConfig(const char *cfgFileName)
{
  // Read the file. If there is an error, report it and exit.
  try {
    std::cout << "Reading config file " << cfgFileName << std::endl;
    cfg.readFile(cfgFileName);

    std::cout.precision(6);
    std::cout << "Settings:" << std::endl;
    
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
    strcpy(file_format, (const char *) cfg.lookup("simulation.file_format"));
    std::cout << "Output file format: " << file_format << std::endl;

  }
  catch(const FileIOException &fioex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const SettingNotFoundException &nfex) {
    std::cerr << "Setting " << nfex.getPath() << " not found." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const SettingTypeException &stex) {
    std::cerr << "Setting type exception." << std::endl;
    return(EXIT_FAILURE);
  }

  std::cout << std::endl;

  return 0;
}


