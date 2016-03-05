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

using namespace libconfig;


/** \file simLorenz63.cpp 
 *  \ingroup examples
 *  \brief Simulate Lorenz (1963) deterministic flow.
 *
 *  Simulate Lorenz (1963) deterministic flow.
 */


/*
 * Declarations
 */

/** \brief User defined function to get parameters from a cfg file using libconfig. */
void readConfig(const char *cfgFileName);

// Configuration
char caseName[256];       //!< Name of the case to simulate 
char file_format[256];    //!< File format of output ("txt" or "bin")
char resDir[256];         //!< Root directory in which results are written
int dim;                  //!< Dimension of the phase space
double rho;               //!< Rayleigh number
double sigma;             //!< \f$ \sigma \f$ parameter
double beta;             //!< \f$ \beta \f$ parameter
gsl_vector *initState;    //!< Initial state
double LCut;              //!< Length of the time series without spinup
double spinup;            //!< Length of initial spinup period to remove
double L;                 //!< Total length of integration
double dt;                //!< Time step of integration
double printStep;         //!< Time step of output
size_t printStepNum;      //!< Time step of output in number of time steps of integration
char postfix[256];        //!< Postfix of destination files
char dstFileName[256];    //!< Destination file name


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

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63(rho, sigma, beta);

  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim, dt);

  // Define model
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme, initState);
  
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
  char cpyBuffer[256];
  
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
    
    // Get Lorenz63 parameters and update case name
    rho = cfg.lookup("model.rho");
    std::cout << "rho = " << rho << std::endl;
    sigma = cfg.lookup("model.sigma");
    std::cout << "sigma = " << sigma << std::endl;
    beta = cfg.lookup("model.beta");
    std::cout << "beta = " << beta << std::endl;
    strcpy(cpyBuffer, caseName);
    sprintf(caseName, "%s_rho%d_sigma%d_beta%d", cpyBuffer,
	    (int) (rho * 1000), (int) (sigma * 1000), (int) (beta * 1000));
    
    
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

  std::cout << std::endl;

    // Some parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);

    // Define names and open destination file
    sprintf(postfix, "_%s_L%d_spinup%d_dt%d_samp%d", caseName,
	    (int) L, (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)),
	    (int) printStepNum);
    sprintf(dstFileName, "%s/simulation/sim%s.%s", resDir, postfix, file_format);

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

