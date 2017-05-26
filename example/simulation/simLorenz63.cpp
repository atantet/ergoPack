#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simLorenz63.cpp 
 *  \ingroup examples
 *  \brief Simulate Lorenz (1963) deterministic flow.
 *
 *  Simulate Lorenz (1963) deterministic flow.
 */


/** \brief Simulation of Lorenz (1963) deterministic flow.
 *
 *  Simulation of Lorenz (1963) deterministic flow.
 *  After parsing the configuration file,
 *  the vector field of the Lorenz 1963 flow and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  FILE *dstStream;
  gsl_matrix *X;
  char postfix[256], dstFileName[256];

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

  // Open destination file
  // Define names and open destination file
  sprintf(dstFileName, "%s/simulation/sim%s.%s", resDir, postfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63(&p);

  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);

  // Define model
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);
  
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

  return 0;
}
