#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <transferOperator.hpp>
#include <transferSpectrum.hpp>
#include "../cfg/readConfig.hpp"


/** \file spectrum.cpp
 *  \ingroup examples
 *  \brief Get spectrum of transfer operators.
 *   
 *  Get spectrum of transfer operators.
 */


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
    readObservable(&cfg);
    readGrid(&cfg);
    readTransfer(&cfg);
    readSpectrum(&cfg);
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

  // Declarations
  // Transfer
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256],
    maskFileName[256], srcPostfixSim[256], postfix[256], postfixTau[256];
  transferOperator *transferOp;
  double tau;

  // Eigen problem
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
  transferSpectrum *transferSpec;

  
  sprintf(srcPostfixSim, "_%s_mu%04d_alpha%04d_gamma%04d_delta%04d_beta%04d_eps%04d_sep%04d_L%d_spinup%d_dt%d_samp%d", caseName,
	  (int) (p["mu"] * 10000 + 0.1), (int) (p["alpha"] * 10000 + 0.1),
	  (int) (p["gamma"] * 10000 + 0.1), (int) (p["delta"] * 10000 + 0.1),
	  (int) (p["beta"] * 10000 + 0.1), (int) (p["eps"] * 10000 + 0.1),
	  (int) (p["sep"] * 10000 + 0.1),
	  (int) L, (int) spinup, (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  sprintf(postfix, "%s_nTraj%d%s", srcPostfixSim, (int) nTraj, gridPostfix);

  // Scan matrices and distributions for one lag
  for (size_t itau = 0; itau < tauRng->size; itau++) {
    tau = gsl_vector_get(tauRng, itau);
    std::cout << "\nGetting spectrum for a lag of " << tau << std::endl;

    // Get file names
    sprintf(postfixTau, "%s_tau%04d", postfix, (int) (tau * 10000 + 0.1));
    sprintf(forwardTransitionFileName, \
	    "%s/transfer/forwardTransition/forwardTransition%s.crs%s",
	    resDir, postfixTau, fileFormat);
    sprintf(backwardTransitionFileName, \
	    "%s/transfer/backwardTransition/backwardTransition%s.crs%s",
	    resDir, postfixTau, fileFormat);
    sprintf(EigValForwardFileName,
	    "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	    resDir, nev, postfixTau, fileFormat);
    sprintf(EigValForwardFileName,
	    "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	    resDir, nev, postfixTau, fileFormat);
    sprintf(EigVecForwardFileName,
	    "%s/spectrum/eigvec/eigvecForward_nev%d%s.%s",
	    resDir, nev, postfixTau, fileFormat);
    sprintf(EigValBackwardFileName,
	    "%s/spectrum/eigval/eigvalBackward_nev%d%s.%s",
	    resDir, nev, postfixTau, fileFormat);
    sprintf(EigVecBackwardFileName,
	    "%s/spectrum/eigvec/eigvecBackward_nev%d%s.%s",
	    resDir, nev, postfixTau, fileFormat);

    // Read transfer operator
    std::cout << "Reading stationary transfer operator..." << std::endl;
    try
      {
	/** Construct transfer operator without allocating memory
	    to the distributions (only to the mask) ! */
	transferOp = new transferOperator(N);
	    
	// Scan forward transition matrix (this sets NFilled)
	std::cout << "Scanning forward transition matrix from "
		  << forwardTransitionFileName << std::endl;
	transferOp->scanTransition(forwardTransitionFileName,
				   fileFormat);

	// Scan mask for the first lag
	sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
		resDir, postfixTau, fileFormat);
	std::cout << "Scanning mask from "
		  << maskFileName << std::endl;
	transferOp->scanMask(maskFileName, fileFormat);
      
	std::cout << "Nfilled = " << transferOp->getNFilled() << std::endl;

	// Allocate memory to distributions
	transferOp->allocateDist();

	// Scan initial distribution for the first lag
	sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
		resDir, postfixTau, fileFormat);
	std::cout << "Scanning initial distribution from "
		  << initDistFileName << std::endl;
	transferOp->scanInitDist(initDistFileName, fileFormat);
      
      }
    catch (std::exception &ex)
      {
	std::cerr << "Error reading transfer operator: " << ex.what()
		  << std::endl;
	return EXIT_FAILURE;
      }

      
    // Get spectrum
    try
      {
	// Solve eigen value problem with default configuration
	transferSpec = new transferSpectrum(nev, transferOp, &config);

	if (getForwardEigenvectors)
	  {
	    std::cout << "Solving eigen problem for forward transition matrix..."
		      << std::endl;
	    transferSpec->getSpectrumForward();
	    std::cout << "Found "
		      << transferSpec->getNev()
		      << "/" << nev << " eigenvalues." << std::endl;
	  }
	if (getBackwardEigenvectors)
	  {
	    std::cout << "Solving eigen problem for \
backward transition matrix..."
		      << std::endl;
	    transferSpec->getSpectrumBackward();
	    std::cout << "Found "
		      << transferSpec->getNev()
		      << "/" << nev << " eigenvalues." << std::endl;
	  }
	if (getForwardEigenvectors
	    && getBackwardEigenvectors
	    && makeBiorthonormal)
	  {
	    std::cout << "Making set of forward and backward eigenvectors \
biorthonormal..."
		      << std::endl;
	    transferSpec->makeBiorthonormal();
	  }
      }
    catch (std::exception &ex)
      {
	std::cerr << "Error calculating spectrum: " << ex.what() << std::endl;
	return EXIT_FAILURE;
      }
  
    // Write spectrum 
    try
      {
	if (getForwardEigenvectors)
	  {
	    std::cout << "Writing forward eigenvalues and eigenvectors..."
		      << std::endl;
	    transferSpec->writeSpectrumForward(EigValForwardFileName,
					       EigVecForwardFileName,
					       fileFormat);
	  }
	if (getBackwardEigenvectors)
	  {
	    std::cout << "Writing backward eigenvalues and eigenvectors..."
		      << std::endl;
	    transferSpec->writeSpectrumBackward(EigValBackwardFileName,
						EigVecBackwardFileName,
						fileFormat);
	  }
      }
    catch (std::exception &ex)
      {
	std::cerr << "Error writing spectrum: " << ex.what() << std::endl;
	return EXIT_FAILURE;
      }

    // Free
    delete transferSpec;
    delete transferOp;
  }
  freeConfig();
  
return 0;
}
