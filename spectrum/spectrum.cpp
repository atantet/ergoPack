#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <ergoPack/transferOperator.hpp>
#include <ergoPack/transferSpectrum.hpp>
#include "../cfg/readConfig.hpp"


/** \file spectrum.cpp
 *  \ingroup examples
 *  \brief Get spectrum of transfer operators.
 *   
 *  Get spectrum of transfer operators.
 */


// Configuration variables
char resDir[256];               //!< Root directory in which results are written
char caseName[256];             //!< Name of the case to simulate 
char file_format[256];          //!< File format of output ("txt" or "bin")
char delayName[256];            //!< Name associated with the number and values of the delays
int dim;                        //!< Dimension of the phase space
double LCut;                    //!< Length of the time series without spinup
double spinup;                  //!< Length of initial spinup period to remove
double L;                       //!< Total length of integration
double dt;                      //!< Time step of integration
double printStep;               //!< Time step of output
size_t printStepNum;            //!< Time step of output in number of time steps of integration
char srcPostfix[256];           //!< Postfix of simulation file.
char srcFileName[256];          //!< Name of the source simulation file
char dstFileName[256];          //!< Destination file name
size_t nt0;                     //!< Number of time steps of the source time series
size_t nt;                      //!< Number of time steps of the observable
int dimObs;                     //!< Dimension of the observable
size_t embedMax;                //!< Maximum lag for the embedding
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
char gridFileName[256];         //!< File name for the grid file
configAR config;                //!< Configuration data for the eigen problem
char configFileName[256];       //!< Name of the configuration file
bool stationary;                //!< Whether the problem is stationary or not
bool getForwardEigenvectors;    //!< Whether to get forward eigenvectors
bool getBackwardEigenvectors;   //!< Whether to get backward eigenvectors
bool makeBiorthonormal;         //!< Whether to make eigenvectors biorthonormal


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
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt",
	      resDir, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigvalForward_nev%d%s.txt",
	      resDir, nev, postfix);
      sprintf(finalDistFileName, "%s/transfer/finalDist/finalDist%s.txt",
	      resDir, postfix);
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
	  transferOp->scanInitDist(initDistFileName);
	  transferOp->scanForwardTransition(forwardTransitionFileName);

	  if (!stationary)
	    {
	      transferOp->scanFinalDist(finalDistFileName);
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
  
  return 0;
}
