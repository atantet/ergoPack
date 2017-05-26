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
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simOU.cpp 
 *  \ingroup examples
 *  \brief Simulate a multi-dimensional Ornstein-Uhlenbeck process.
 *
 *  Simulate a multi-dimensional Ornstein-Uhlenbeck process.
 */


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
  if (argc < 2) {
    std::cout << "Enter path to configuration file:" << std::endl;
    std::cin >> configFileName;
  }
  else
    strcpy(configFileName, argv[1]);
  try {
    Config cfg;
    std::cout << "Sparsing config file " << configFileName << std::endl;
    cfg.readFile(configFileName);
    readGeneral(&cfg);
    readModel(&cfg);
    readSimulation(&cfg);
    readSprinkle(&cfg);
    std::cout << "Sparsing success.\n" << std::endl;
  }
  catch(const SettingTypeException &ex) {
    std::cerr << "Setting " << ex.getPath() << " type exception."
	      << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "Setting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "Setting " << ex.getPath() << " name exception."
	      << std::endl;
    throw ex;
  }
  catch(const ParseException &ex) {
    std::cerr << "Parse error at " << ex.getFile() << ":" << ex.getLine()
              << " - " << ex.getError() << std::endl;
    throw ex;
  }
  catch(const FileIOException &ex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw ex;
  }
  catch (...) {
    std::cerr << "Error reading configuration file" << std::endl;
    return(EXIT_FAILURE);
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
  numericalSchemeStochastic *scheme = new EulerMaruyama(dim);

  // Define model
  std::cout << "Defining stochastic model..." << std::endl;
  modelStochastic *mod = new modelStochastic(field, stocField, scheme);
  
  // Numerical integration
  printf("Integrating simulation...\n");
  X = gsl_matrix_alloc(1, 1); // Fake allocation
  mod->integrate(initState, L, dt, spinup, printStepNum, &X);

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
  freeConfig();

  return 0;
}


