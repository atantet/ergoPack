#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <libconfig.h++>
#include <ergoPack/transferOperator.hpp>
#include <ergoPack/transferSpectrum.hpp>


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
void readConfig(const char *cfgFileName);

// Paths
const char resDir[] = "../results/";

// Configuration 
Config cfg;
char caseName[256];
double LCut, L, dt, spinup;
double printStep;
size_t printStepNum;
int dimObs;
gsl_vector_uint *components;
gsl_vector_uint *embedding;
size_t N;
gsl_vector_uint *nx;
gsl_vector *nSTDLow, *nSTDHigh;
size_t nLags;
gsl_vector *tauRng;
int nev;
// File names
char obsName[256], srcPostfix[256];
char gridPostfix[256], gridCFG[256], gridFileName[256];


// Main program
int main(int argc, char * argv[])
{
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

  
  // Declarations
  // Transfer
  double tau;
  char forwardTransitionFileName[256], backwardTransitionFileName[256],
    initDistFileName[256], finalDistFileName[256], postfix[256];
  transferOperator *transferOp;

  // Eigen problem
  int nconv;
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
  transferSpectrum *transferSpec;

  
  // Scan matrices and distributions for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);

      // Get file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));
      sprintf(forwardTransitionFileName, \
	      "%s/transfer/forwardTransition%s.coo", resDir, postfix);
      sprintf(backwardTransitionFileName,
	      "%s/transfer/backwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transfer/initDist%s.txt",
	      resDir, postfix);
      sprintf(finalDistFileName, "%s/transfer/finalDist%s.txt",
	      resDir, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigval_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecForwardFileName, "%s/spectrum/eigvec/eigvec_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigValBackwardFileName, "%s/spectrum/eigval/eigvalAdjoint_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecBackwardFileName, "%s/spectrum/eigvec/eigvecAdjoint_nev%d%s.txt",
	      resDir, nev, postfix);

      
      /** Read transfer operator */
      std::cout << "Reading transfer operator..." << std::endl;
      try
	{
	  transferOp = new transferOperator(N);
	  transferOp->scanInitDist(initDistFileName);
	  transferOp->scanFinalDist(finalDistFileName);
	  transferOp->scanForwardTransition(forwardTransitionFileName);
	  transferOp->scanBackwardTransition(backwardTransitionFileName);
	}
      catch (std::exception &ex)
	{
	  std::cerr << "Error reading transfer operator: " << ex.what() << std::endl;
	  return EXIT_FAILURE;
	}

      
      /** Get spectrum */
      try
	{
	  // Solve eigen value problem with default configuration
	  std::cout << "Solving eigen problem" << std::endl;
	  transferSpec = new transferSpectrum(nev, transferOp);
	  nconv = transferSpec->getSpectrum();
	  std::cout << "Found " << nconv << "/" << (nev * 2) << " eigenvalues." << std::endl;
	}
      catch (std::exception &ex)
	{
	  std::cerr << "Error calculating spectrum: " << ex.what() << std::endl;
	  return EXIT_FAILURE;
	}
  
      /** Write spectrum */
      try
	{
	  std::cout << "Write spectrum..." << std::endl;
	  nconv = transferSpec->writeSpectrum(EigValForwardFileName, EigVecForwardFileName,
					      EigValBackwardFileName, EigVecBackwardFileName);
	}
      catch (std::exception &ex)
	{
	  std::cerr << "Error writing spectrum: " << ex.what() << std::endl;
	  return EXIT_FAILURE;
	}

      // Free
      delete transferOp;
      delete transferSpec;
  }

  // Free
  gsl_vector_uint_free(components);
  gsl_vector_uint_free(embedding);
  gsl_vector_uint_free(nx);
  gsl_vector_free(nSTDLow);
  gsl_vector_free(nSTDHigh);
  gsl_vector_free(tauRng);
  
  return 0;
}


// Definitions
void
readConfig(const char *cfgFileName)
{
  char cpyBuffer[256];
  
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

    
    // Get observable settings
    std::cout << std::endl << "---observable---" << std::endl;
    
    // Components
    const Setting &compSetting = cfg.lookup("observable.components");
    dimObs = compSetting.getLength();
    components = gsl_vector_uint_alloc(dimObs);
    std::cout << "Components: [";
    for (size_t d = 0; d < (size_t) dimObs; d++)
      {
	gsl_vector_uint_set(components, d, compSetting[d]);
	std::cout << gsl_vector_uint_get(components, d) << " ";
      }
    std::cout << "]" << std::endl;

    // Embedding
    const Setting &embedSetting = cfg.lookup("observable.embeddingDays");
    embedding = gsl_vector_uint_alloc(dimObs);
    sprintf(obsName, "");
    std::cout << "Embedding: [";
    for (size_t d = 0; d < (size_t) dimObs; d++)
      {
	double embd = embedSetting[d];
	gsl_vector_uint_set(embedding, d, embd / 365 / printStep);
	std::cout << gsl_vector_uint_get(embedding, d) << " ";
	sprintf(cpyBuffer, obsName);
	sprintf(obsName, "%s_c%zu_e%d", cpyBuffer,
		gsl_vector_uint_get(components, d), (int) embd);
      }
    std::cout << "]" << std::endl;
			    
    
    // Get grid settings
    std::cout << std::endl << "---grid---" << std::endl;
    const Setting &nxSetting = cfg.lookup("grid.nx");
    const Setting &nSTDLowSetting = cfg.lookup("grid.nSTDLow");
    const Setting &nSTDHighSetting = cfg.lookup("grid.nSTDHigh");
    nx = gsl_vector_uint_alloc(dimObs);
    nSTDLow = gsl_vector_alloc(dimObs);
    nSTDHigh = gsl_vector_alloc(dimObs);
    N = 1;
    for (size_t d = 0; d < (size_t) (dimObs); d++)
      {
	gsl_vector_uint_set(nx, d, nxSetting[d]);
	N *= gsl_vector_uint_get(nx, d);
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
    std::cout << std::endl << "---spectrum---" << std::endl
	      << "nev: " << nev << std::endl;
    
    std::cout << std::endl;

  }
  catch(const SettingNotFoundException &nfex) {
    std::cerr << "Setting " << nfex.getPath() << " not found." << std::endl;
    throw nfex;
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


  // Finish configuration
  // Define time series parameters
  L = LCut + spinup;
  printStepNum = (size_t) (printStep / dt);

  // Define postfix and src file name
  sprintf(srcPostfix, "_%s_L%d_spinup%d_dt%d_samp%d", caseName,
	  (int) L, (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)),
	  (int) printStepNum);

  // Define grid name
  sprintf(gridCFG, "");
  for (size_t d = 0; d < (size_t) dimObs; d++) {
    strcpy(cpyBuffer, gridCFG);
    sprintf(gridCFG, "%s_n%dl%dh%d", cpyBuffer,
	    gsl_vector_uint_get(nx, d),
	    (int) gsl_vector_get(nSTDLow, d),
	    (int) gsl_vector_get(nSTDHigh, d));
  }
  sprintf(gridPostfix, "%s%s%s", srcPostfix, obsName, gridCFG);
  sprintf(gridFileName, "%s/grid/grid%s.txt", resDir, gridPostfix);


  return;
}
