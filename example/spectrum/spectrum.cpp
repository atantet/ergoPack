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
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256], postfix[256];
  transferOperator *transferOp;
  gsl_vector *initDist = gsl_vector_alloc(N);
  gsl_vector *finalDist = gsl_vector_alloc(N);


  // Eigen problem
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
  transferSpectrum *transferSpec;

  
  // Scan matrices and distributions for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);
      std::cout << "\nGetting spectrum for a lag of " << tau << std::endl;

      // Get file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));
      sprintf(forwardTransitionFileName, \
	      "%s/transfer/forwardTransition/forwardTransition%s.coo", resDir, postfix);
      sprintf(backwardTransitionFileName, \
	      "%s/transfer/backwardTransition/backwardTransition%s.coo", resDir, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigvalForward_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigvalForward_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecForwardFileName, "%s/spectrum/eigvec/eigvecForward_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigValBackwardFileName, "%s/spectrum/eigval/eigvalBackward_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(EigVecBackwardFileName, "%s/spectrum/eigvec/eigvecBackward_nev%d%s.txt",
	      resDir, nev, postfix);

      // Read transfer operator
      std::cout << "Reading stationary transfer operator..." << std::endl;
      try
	{
	  transferOp = new transferOperator(N, stationary);
	    
	  // Only scan initial distribution for the first lag
	  if (lag == 0)
	    {
	      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt",
		      resDir, gridPostfix);
	      std::cout << "Scanning initial distribution from " << initDistFileName << std::endl;
	      transferOp->scanInitDist(initDistFileName);
	      gsl_vector_memcpy(initDist, transferOp->initDist);
	    }
	  else
	    transferOp->setInitDist(initDist);
	  // Scan forward transition matrix
	  std::cout << "Scanning forward transition matrix from "
		    << forwardTransitionFileName << std::endl;
	  transferOp->scanForwardTransition(forwardTransitionFileName);

	  if (!stationary)
	    {
	      // Only scan final distribution for the first lag
	      if (lag == 0)
		{
		  sprintf(finalDistFileName, "%s/transfer/finalDist/finalDist%s.txt",
			  resDir, gridPostfix);
		  std::cout << "Scanning final distribution from "
			    << finalDistFileName << std::endl;
		  transferOp->scanFinalDist(finalDistFileName);
		  gsl_vector_memcpy(finalDist, transferOp->finalDist);
		}
	      else
		transferOp->setFinalDist(finalDist);
	      // Scan backward transition matrix
	      std::cout << "Scanning backward transition matrix from "
			<< backwardTransitionFileName << std::endl;
	      transferOp->scanBackwardTransition(backwardTransitionFileName);
	    }
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

	  if (getForwardEigenvectors)
	    {
	      std::cout << "Solving eigen problem for forward transition matrix..." << std::endl;
	      transferSpec->getSpectrumForward();
	      std::cout << "Found " << transferSpec->EigProbForward.ConvergedEigenvalues()
			<< "/" << nev << " eigenvalues." << std::endl;	      
	    }
	  if (getBackwardEigenvectors)
	    {
	      std::cout << "Solving eigen problem for backward transition matrix..." << std::endl;
	      transferSpec->getSpectrumBackward();
	      std::cout << "Found " << transferSpec->EigProbBackward.ConvergedEigenvalues()
			<< "/" << nev << " eigenvalues." << std::endl;
	    }
	  if (makeBiorthonormal)
	    {
	      std::cout << "Making set of forward and backward eigenvectors biorthonormal..."
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
	      std::cout << "Writing forward eigenvalues and eigenvectors..." << std::endl;
	      transferSpec->writeSpectrumForward(EigValForwardFileName,
						 EigVecForwardFileName);
	    }
	  if (getBackwardEigenvectors)
	    {
	      std::cout << "Writing backward eigenvalues and eigenvectors..." << std::endl;
	      transferSpec->writeSpectrumBackward(EigValBackwardFileName,
						 EigVecBackwardFileName);
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

  // Free
  freeConfig();
  gsl_vector_free(initDist);
  gsl_vector_free(finalDist);
  
  return 0;
}
