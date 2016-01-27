#define DIM 2
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <libconfig.h++>
#include <ATSuite/transferOperator.hpp>
#include <ATSuite/transferSpectrum.hpp>


using namespace libconfig;


/** \file spectrum.cpp
 *  \brief Get spectrum of transfer operators.
 *   
 * Get spectrum of transfer operators.
 * The configuration file given as first command-line argument
 * is parsed using libconfig C++ library.
 * The transition matrices are then read from matrix files in coordinate format.
 * The Eigen problem is then defined and solved using ARPACK++.
 * Finally, the results are written to file.
 */


// Declarations
/** \brief User defined function to get parameters from a cfg file using libconfig. */
int readConfig(const char *cfgFileName);

// Paths
const char simDir[] = "../results/simulations/";
const char resDir[] = "../results/";

// Configuration 
Config cfg;
char caseName[256], file_format[256];
int dim;
double LCut, L, dt, spinup;
double printStep;
size_t printStepNum;
int dimEmbed;
gsl_vector_uint *nx;
gsl_vector *nSTDLow, *nSTDHigh;
size_t nLags;
gsl_vector *tauRng;
int nev;
int minNumberStates;


// Main program
int main(int argc, char * argv[])
{
  // Read configuration file
  if (readConfig("transferCZ")) {
    std::cerr << "Error reading config file " << "transferCZ" << "."
	      << std::endl;
    return(EXIT_FAILURE);
  }

  
  // Declarations
  // Observable
  char obsName[256], srcPostfix[256], cpyBuffer[256], postfix[256];
  
  // Grid
  size_t N = 1;
  char gridPostfix[256];

  // Transfer
  double tauDim;
  char forwardTransitionFileName[256], backwardTransitionFileName[256],
    initDistFileName[256], finalDistFileName[256];
  transferOperator *transferOp;

  // Filtering
  double tol;

  // Eigen problem
  int nconv;
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
  transferSpectrum *transferSpec;

  
  // Define grid name
  sprintf(gridCFG, "");
  for (size_t d = 0; d < (size_t) dimEmbed; d++) {
    strcpy(cpyBuffer, gridCFG);
    sprintf(gridCFG, "%s_n%dl%dh%d", cpyBuffer,
	    gsl_vector_uint_get(nx, d),
	    (int) gsl_vector_get(nSTDLow, d),
	    (int) gsl_vector_get(nSTDHigh, d));
  }
  sprintf(gridPostfix, "_%s%s%s%d%s",
	  srcPostfix, obsName, sEmbed, dimEmbed, gridCFG);
  sprintf(gridFileName, "%s/grid/grid%s.txt", resDir, gridPostfix);

  
  // Scan matrices and distributions for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);
      tol = minNumberStates * 1. / (nt - tauDim);
      std::cout << "alpha = " << tol << std::endl;

      // Get file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tauDim * 1000));
      sprintf(forwardTransitionFileName, \
	      "%s/transitionMatrix/forwardTransition%s.coo", resDir, postfix);
      sprintf(backwardTransitionFileName,
	      "%s/transitionMatrix/backwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transitionMatrix/initDist%s.txt",
	      resDir, postfix);
      sprintf(finalDistFileName, "%s/transitionMatrix/finalDist%s.txt",
	      resDir, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigval_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecForwardFileName, "%s/spectrum/eigvec/eigvec_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigValBackwardFileName, "%s/spectrum/eigval/eigvalAdjoint_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecBackwardFileName, "%s/spectrum/eigvec/eigvecAdjoint_nev%d%s.txt",
	      resDir, nev, postfix);

      // Read transfer operator
      std::cout << "Reading transfer operator..." << std::endl;
      transferOp = new transferOperator;
      transferOp->N = N;
      transferOp->scanInitDist(initDistFileName);
      transferOp->scanFinalDist(finalDistFileName);
      transferOp->scanForwardTransition(forwardTransitionFileName);
      transferOp->scanBackwardTransition(backwardTransitionFileName);

      // Filter and get left stochastic
      //transferOp->filter(tol);

      // Solve eigen value problem with default configuration
      std::cout << "Solving eigen problem for the first " << nev << std::endl;
      transferSpec = new transferSpectrum(nev, transferOp);
      nconv = transferSpec->getSpectrum();
      std::cout << "Found " << nconv << "/" << (nev * 2) << " eigenvalues." << std::endl;

      // Open destination files and write spectrum
      std::cout << "Write spectrum..." << std::endl;
      nconv = transferSpec->writeSpectrum(EigValForwardFileName, EigVecForwardFileName,
					  EigValBackwardFileName, EigVecBackwardFileName);

      // Free
      delete transferOp;
      delete transferSpec;
  }
  
  return 0;
}


// Definitions
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
    
    /** Get simulation settings */
    std::cout << "\n" << "---simulation---" << std::endl;

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

    
    // Get embedding settings
    if (exists("embedding.dimEmbed"))
      {
	dimEmbed = cfg.lookup("embedding.dimEmbed");
	std::cout << std::endl << "---embedding---" << std::endl
		  << "dimEmbed: " << dimEmbed << std::endl;
      }
    else
      dimEmbed = 1;

    
    // Get grid settings
    std::cout << std::endl << "---grid---" << std::endl;
    const Setting &nxSetting = cfg.lookup("grid.nx");
    const Setting &nSTDLowSetting = cfg.lookup("grid.nSTDLow");
    const Setting &nSTDHighSetting = cfg.lookup("grid.nSTDHigh");
    nx = gsl_vector_uint_alloc(dim * dimEmbed);
    nSTDLow = gsl_vector_alloc(dim * dimEmbed);
    nSTDHigh = gsl_vector_alloc(dim * dimEmbed);
    for (size_t d = 0; d < (size_t) (dim * dimEmbed); d++) {
      gsl_vector_uint_set(nx, d, nxSetting[d]);
      gsl_vector_set(nSTDLow, d, nSTDLowSetting[d]);
      gsl_vector_set(nSTDHigh, d, nSTDHighSetting[d]);
      std::cout << "Grid definition (nSTDLow, nSTDHigh, n):" << std::endl;
      std::cout << "dim " << d+1 << ": ("
		<< gsl_vector_get(nSTDLow, d) << ", "
		<< gsl_vector_get(nSTDHigh, d) << ", "
		<< gsl_vector_uint_get(nx, d) << ")" << std::endl;
    }


    // Get transition settings
    const Setting &tauRngSetting = cfg.lookup("transfer.tauRng");
    nLags = tauRngSetting.getLength();
    tauRng = gsl_vector_alloc(nLags);

    std::cout << std::endl << "---transfer---" << std::endl;
    std::cout << "tauRng = [";
    for (size_t lag = 0; lag < nLags; lag++) {
      gsl_vector_set(tauRng, lag, tauRngSetting[lag]);
      std::cout << gsl_vector_get(tauRng, lag) << " ";
    }
    std::cout << "]" << std::endl;

    // Get spectrum setting 
    nev = cfg.lookup("spectrum.nev");
    minNumberStates = cfg.lookup("spectrum.minNumberStates");
    std::cout << std::endl << "---spectrum---" << std::endl
	      << "nev: " << nev << std::endl
	      << "minNumberStates: " << minNumberStates << std::endl;
    
    std::cout << std::endl;

  }
  catch(const SettingNotFoundException &nfex) {
    std::cerr << "Setting " << nfex.getPath() << " not found." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const FileIOException &fioex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return(EXIT_FAILURE);
    std::cout << "Settings:" << std::endl;
  }

  return 0;
}
