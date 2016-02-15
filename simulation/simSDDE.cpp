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
#include <ergoPack/ODESolvers.hpp>
#include <ergoPack/SDESolvers.hpp>
#include <ergoPack/SDDESolvers.hpp>

using namespace libconfig;


/** \file simSDDE.cpp 
 *  \ingroup examples
 *  \brief Simulate a Stochastic Delay Differential Equation.
 *
 *  Simulate a Stochastic Delay Differential Equation
 *  with polynomial vector fields.
 */

// Declarations
/** \brief User defined function to get parameters from a cfg file using libconfig. */
void readConfig(const char *cfgFileName);

char caseName[256];       //!< Name of the case to simulate 
char file_format[256];    //!< File format of output ("txt" or "bin")
char resDir[256];         //!< Root directory in which results are written
int dim;                  //!< Dimension of the phase space
size_t nDelays;           //!< Number of delays
/** Vector of polynomials to define the vector fields for each delay */
std::vector<gsl_vector *> *driftPolynomials;
gsl_vector *delaysDays;   //!< Delays in days
gsl_vector_uint *delays;  //!< Delays in number of time steps
char delayName[256];      //!< Name associated with the delays
gsl_matrix *Q;            //!< Diffusion matrix
gsl_vector *initStateCst; //!< State to define constant initial state history
double LCut;              //!< Length of the time series without spinup
double spinup;            //!< Length of initial spinup period to remove
double L;                 //!< Total length of integration
double dt;                //!< Time step of integration
double printStep;         //!< Time step of output
size_t printStepNum;      //!< Time step of output in number of time steps of integration
char postfix[256];        //!< Postfix of destination files
char dstFileName[256];    //!< Destination file name


/** \brief Simulation of an Stochastic Delay Differential Equation.
 *
 *  Simulation of an Stochastic Delay Differential Equation.
 *  After parsing the configuration file,
 *  a delayed polynomial vector field for the drift, a diffusion matrix
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
      return EXIT_FAILURE;
    }

  // Open destination file
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

  // Define delayed field from vector fields for each delay
  std::cout << "Defining delayed vector field..." << std::endl;
  std::vector<vectorField *> *fields = new std::vector<vectorField *>(nDelays);
  for (size_t d = 0; d < nDelays; d++)
    fields->at(d) = new polynomial1D(driftPolynomials->at(d));
  vectorFieldDelay *delayedField = new vectorFieldDelay(fields, delays);

  // Define stochastic vector field
  std::cout << "Defining stochastic vector field..." << std::endl;
  vectorFieldStochastic *stocField = new additiveWiener(Q, r);

  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalSchemeSDDE *scheme = new EulerMaruyamaSDDE(dim, nDelays, dt);

  // Define model with zero initial state
  std::cout << "Defining model..." << std::endl;
  modelSDDE *mod = new modelSDDE(delayedField, stocField, scheme, initStateCst);

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
  gsl_vector_free(initStateCst);
  gsl_vector_free(delaysDays);
  gsl_vector_uint_free(delays);
  for (size_t d = 0; d < nDelays; d++)
      gsl_vector_free(driftPolynomials->at(d));
  delete driftPolynomials;
  delete delayedField;
  delete stocField;
  delete scheme;
  delete mod;
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
  size_t degree;
  char cpyBuffer[256];

  // Read the file. If there is an error, report it and exit.
  std::cout << "Reading config file " << cfgFileName << std::endl;
  try {
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

    // Get delays in days and the number of delays
    const Setting &delaysSetting = cfg.lookup("model.delaysDays");
    nDelays = (size_t) delaysSetting.getLength();
    delaysDays = gsl_vector_alloc(nDelays);
    std::cout << "delays (days) = [";
    sprintf(delayName, "");
    for (size_t d = 0; d < nDelays; d++)
      {
	gsl_vector_set(delaysDays, d, delaysSetting[d]);
	std::cout << gsl_vector_get(delaysDays, d) << " ";
	strcpy(cpyBuffer, delayName);
	sprintf(delayName, "%s_d%d", cpyBuffer, (int) gsl_vector_get(delaysDays, d));
      }
    std::cout << "]" << std::endl;

    // Get drift polynomials
    const Setting &driftSetting = cfg.lookup("model.drift");
    driftPolynomials = new std::vector<gsl_vector *>(nDelays);
    for (size_t d = 0; d < nDelays; d++)
      {
	degree = driftSetting[d].getLength() - 1;
	driftPolynomials->at(d) = gsl_vector_alloc(degree + 1);
	// Print matrices
	std::cout << "Field polynomial coefficients at delay "
		  << d << " = [";
	for (size_t c = 0; c < degree + 1; c++)
	  {
	    gsl_vector_set(driftPolynomials->at(d), c, driftSetting[d][c]);
	    std::cout << gsl_vector_get(driftPolynomials->at(d), c) << " ";
	  }
	std::cout << "]" << std::endl;
      }

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
    const Setting &initStateSetting = cfg.lookup("simulation.initStateCst");
    initStateCst = gsl_vector_alloc(dim);
    std::cout << "Constant initial state: [";
    for (size_t i = 0; i < (size_t) dim; i++)
      {
	gsl_vector_set(initStateCst, i, initStateSetting[i]);
	std::cout << gsl_vector_get(initStateCst, i) << " ";
      }
    std::cout << "]" << std::endl;

    // Simulation length without spinup
    LCut = cfg.lookup("simulation.LCut");
    std::cout << "LCut = " << LCut << std::endl;

    // Time step
    dt = cfg.lookup("simulation.dt");
    std::cout << "dt = " << dt << std::endl;

    // Get delays in time-steps
    delays = gsl_vector_uint_alloc(nDelays);
    for (size_t d = 0; d < nDelays; d++)
      {
	gsl_vector_uint_set(delays, d,
			    (unsigned int) (gsl_vector_get(delaysDays, d) / 365 / dt));
      }

    // Spinup period to remove
    spinup = cfg.lookup("simulation.spinup");
    std::cout << "spinup = " << spinup << std::endl;

    // Sub-sampling for recording
    printStep = cfg.lookup("simulation.printStep");
    std::cout << "printStep = " << printStep << std::endl;

    // Output format
    strcpy(file_format, (const char *) cfg.lookup("simulation.file_format"));
    std::cout << "Output file format: " << file_format << std::endl;

    std::cout << std::endl;
    
    // Define some parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);
    
    // Define destination file name
    sprintf(postfix, "_%s%s_L%d_spinup%d_dt%d_samp%d", caseName, delayName,
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
