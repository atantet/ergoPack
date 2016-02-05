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
 */


// Declarations
/** \brief Sparse configuration file using libconfig++. */
void readConfig(const char *cfgFileName);

// Configuration 
char resDir[256];               //!< Root directory in which results are written
char caseName[256];             //!< Name of the case to simulate 
char delayName[256];            //!< Name associated with the number and values of the delays
double LCut;                    //!< Length of the time series without spinup
double spinup;                  //!< Length of initial spinup period to remove
double L;                       //!< Total length of integration
double dt;                      //!< Time step of integration
double printStep;               //!< Time step of output
size_t printStepNum;            //!< Time step of output in number of time steps of integration
char srcPostfix[256];           //!< Postfix of simulation file.
char dstFileName[256];          //!< Destination file name
int dimObs;                     //!< Dimension of the observable
gsl_vector_uint *components;    //!< Components in the time series used by the observable
gsl_vector_uint *embedding;     //!< Embedding lags for each component
size_t N;                       //!< Dimension of the grid
gsl_vector_uint *nx;            //!< Number of grid boxes per dimension
gsl_vector *nSTDLow;            //!< Number of standard deviations below mean to span by the grid 
gsl_vector *nSTDHigh;           //!< Number of standard deviations above mean to span by the grid 
size_t nLags;                   //!< Number of transition lags for which to calculate the spectrum
gsl_vector *tauRng;             //!< Lags for which to calculate the spectrum
int nev;                        //!< Number of eigenvectors to calculate
char obsName[256];              //!< Name associated with the observable
char gridPostfix[256];          //!< Postfix associated with the grid
configAR config;                //!< Configuration data for the eigen problem
char configFileName[256];       //!< Name of the configuration file


/** \brief Calculate the spectrum of a transfer operator.
 * 
 * After parsing the configuration file,
 * the transition matrices are then read from matrix files in coordinate format.
 * The Eigen problem is then defined and solved using ARPACK++.
 * Finally, the results are written to file.
 */
int main(int argc, char * argv[])
{
  // Read configuration file
  if (argc < 2)
    {
      std::cout << "Enter path to configuration file:" << std::endl;
      std::cin >> configFileName;
    }
  else
    {
      strcpy(configFileName, argv[1]);
    }
  try
   {
     readConfig(configFileName);
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
	      "%s/transfer/forwardTransition/forwardTransition%s.coo", resDir, postfix);
      sprintf(backwardTransitionFileName,
	      "%s/transfer/backwardTransition/backwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt",
	      resDir, postfix);
      sprintf(finalDistFileName, "%s/transfer/finalDist/finalDist%s.txt",
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

      
      // Get spectrum
      try
	{
	  // Solve eigen value problem with default configuration
	  transferSpec = new transferSpectrum(nev, transferOp, config);
	  std::cout << "Solving eigen problem for forward transition matrix..." << std::endl;
	  transferSpec->getSpectrumForward();
	  std::cout << "Found " << transferSpec->EigProbForward.ConvergedEigenvalues()
		    << "/" << nev << " eigenvalues." << std::endl;
	  std::cout << "Solving eigen problem for backward transition matrix..." << std::endl;
	  transferSpec->getSpectrumBackward();
	  std::cout << "Found " << transferSpec->EigProbBackward.ConvergedEigenvalues()
		    << "/" << nev << " eigenvalues." << std::endl;
	}
      catch (std::exception &ex)
	{
	  std::cerr << "Error calculating spectrum: " << ex.what() << std::endl;
	  return EXIT_FAILURE;
	}
  
      // Write spectrum 
      try
	{
	  std::cout << "Write spectrum..." << std::endl;
	  transferSpec->writeSpectrum(EigValForwardFileName, EigVecForwardFileName,
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
    
    // Get delays in days and the number of delays
    sprintf(delayName, "");
    if (cfg.exists("model.delaysDays"))
      {
	const Setting &delaysSetting = cfg.lookup("model.delaysDays");
	std::cout << "Delays (days): [";
	for (int d = 0; d < delaysSetting.getLength(); d++)
	  {
	    double delay = delaysSetting[d];
	    std::cout << delay << " ";
	    strcpy(cpyBuffer, delayName);
	    sprintf(delayName, "%s_d%.0f", cpyBuffer, delay);
	  }
	std::cout << "]" << std::endl;
      }

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
    std::cout << std::endl << "---spectrum---" << std::endl;
    // Get eigen problem configuration
    config = defaultCfgAR;
    if (cfg.exists("spectrum.which"))
      {
	strcpy(config.which, (const char *) cfg.lookup("spectrum.which"));
      }
    if (cfg.exists("spectrum.ncv"))
      {
	config.ncv = cfg.lookup("spectrum.ncv");
      }
    if (cfg.exists("spectrum.tol"))
      {
	config.tol = cfg.lookup("spectrum.tol");
      }
    if (cfg.exists("spectrum.maxit"))
	{
	  config.maxit = cfg.lookup("spectrum.maxit");
	}
    if (cfg.exists("spectrum.AutoShift"))
	{
	  config.AutoShift = (bool) cfg.lookup("spectrum.AutoShift");
	}
    std::cout << "nev: " << nev << std::endl;
    std::cout << "which: " << config.which << std::endl;
    std::cout << "ncv: " << config.ncv << std::endl;
    std::cout << "tol: " << config.tol << std::endl;
    std::cout << "maxit: " << config.maxit << std::endl;
    std::cout << "AutoShift: " << config.AutoShift << std::endl;
    std::cout << std::endl;

    // Finish configuration
    // Define time series parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);

    // Define postfix and src file name
    sprintf(srcPostfix, "_%s%s_L%d_spinup%d_dt%d_samp%d", caseName, delayName,
	    (int) L, (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)),
	    (int) printStepNum);

    // Define grid name
    sprintf(gridPostfix, "");
    for (size_t d = 0; d < (size_t) dimObs; d++) {
      strcpy(cpyBuffer, gridPostfix);
      sprintf(gridPostfix, "%s_n%dl%dh%d", cpyBuffer,
	      gsl_vector_uint_get(nx, d),
	      (int) gsl_vector_get(nSTDLow, d),
	      (int) gsl_vector_get(nSTDHigh, d));
    }
    strcpy(cpyBuffer, gridPostfix);    
    sprintf(gridPostfix, "%s%s%s", srcPostfix, obsName, cpyBuffer);


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
  catch(const SettingTypeException &stex) {
    std::cerr << "Setting type exception." << std::endl;
    throw stex;
  }


  return;
}
