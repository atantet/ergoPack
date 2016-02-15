#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include "../cfg/readConfig.hpp"
#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <Epetra_RowMatrixTransposer.h>
#include <EpetraExt_MultiVectorOut.h>
#include <AnasaziEpetraAdapter.hpp>
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>
#include <AnasaziBasicOutputManager.hpp>
#include "Teuchos_LAPACK.hpp"


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
bool readGridMem;               //!< Whether to read the grid membership vector
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
  typedef double ScalarType;
  typedef Teuchos::ScalarTraits<ScalarType>          SCT;
   typedef SCT::magnitudeType               MagnitudeType;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<ScalarType, MV> MVT;


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
  char forwardTransitionFileName[256], backwardTransitionFileName[256],
    initDistFileName[256], postfix[256];
  gsl_vector *initDist;

  // Eigen problem
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
  bool success;

  
  // Create an Anasazi output manager
  Anasazi::BasicOutputManager<ScalarType> printer;
  std::string which ("LM");
  int blockSize = 1;
  int numBlocks = nev * 6;
  int numRestarts = 100;

  // Create an Epetra communicator
  Epetra_SerialComm Comm;
      
  // Dimension of the matrix
  //
  // Size of matrix nx*nx
  const int NumGlobalElements = N;

  // Construct a Map that puts approximately the same number of
  // equations on each process.
  std::cout << "Constructing a map for " << NumGlobalElements << " elements..." << std::endl;
  Epetra_Map Map (NumGlobalElements, 0, Comm);
  
  // Create a sort manager to pass into the block Krylov-Schur solver manager
  // -->  Make sure the reference-counted pointer is of type Anasazi::SortManager<>
  // -->  The block Krylov-Schur solver manager uses Anasazi::BasicSort<> by default,
  //      so you can also pass in the parameter "Which", instead of a sort manager.
  Teuchos::RCP<Anasazi::SortManager<MagnitudeType> > MySort =
    Teuchos::rcp( new Anasazi::BasicSort<MagnitudeType>( which ) );

  // Create parameter list to pass into the solver manager
  //
  Teuchos::ParameterList MyPL;
  MyPL.set( "Sort Manager", MySort );
  MyPL.set ("Block Size", blockSize);
  MyPL.set( "Num Blocks", numBlocks );
  MyPL.set ("Maximum Iterations", config.maxit);
  MyPL.set( "Maximum Restarts", numRestarts);
  MyPL.set ("Convergence Tolerance", config.tol);
  MyPL.set ("Orthogonalization", "TSQR");
  MyPL.set ("Verbosity", Anasazi::FinalSummary);

  // Scan matrices and distributions for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      double tau = gsl_vector_get(tauRng, lag);
      Epetra_CrsMatrix *PT, *QT;
  
      std::cout << "\nGetting spectrum for a lag of " << tau << std::endl;

      // Get file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));
      sprintf(forwardTransitionFileName, \
	      "%s/transfer/forwardTransition/forwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt",
	      resDir, postfix);
      sprintf(EigValForwardFileName, "%s/spectrum/eigval/eigvalForward_nev%d%s.txt",
      	      resDir, nev, postfix);
      sprintf(EigVecForwardFileName, "%s/spectrum/eigvec/eigvecForward_nev%d%s.txt",
      	      resDir, nev, postfix);
      sprintf(EigValBackwardFileName, "%s/spectrum/eigval/eigvalBackward_nev%d%s.txt",
      	      resDir, nev, postfix);
      sprintf(EigVecBackwardFileName, "%s/spectrum/eigvec/eigvecBackward_nev%d%s.txt",
      	      resDir, nev, postfix);
      // sprintf(EigValForwardFileName, "test/eigvalForward_nev%d%s.txt",
      // 	      nev, postfix);
      // sprintf(EigVecForwardFileName, "test/eigvecForward_nev%d%s.txt",
      // 	      nev, postfix);
      // sprintf(EigValBackwardFileName, "test/eigvalBackward_nev%d%s.txt",
      // 	      nev, postfix);
      // sprintf(EigVecBackwardFileName, "test/eigvecBackward_nev%d%s.txt",
      // 	      nev, postfix);

      // Read stationary distribution
      if (stationary)
	{
	  FILE *fp;
	  if (!(fp = fopen(initDistFileName, "r")))
	    throw std::ios::failure("transferOperator::scanInitDist, \
opening stream to read");
	  initDist = gsl_vector_alloc(N);
	  gsl_vector_fscanf(fp, initDist);
	  fclose(fp);
	}

      
      // Read forward transition matrix
      std::cout << "Reading transition matrix and transpose..." << std::endl;
      EpetraExt::MatrixMarketFileToCrsMatrix(forwardTransitionFileName, Map, PT,
					     true);
      Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcpFromRef(*PT);

      // Create an Epetra_MultiVector for an initial vector to start the
      // solver.  Note: This needs to have the same number of columns as
      // the blocksize.
      Teuchos::RCP<Epetra_MultiVector> ivec
	= Teuchos::rcp (new Epetra_MultiVector (Map, blockSize));
      ivec->Random (); // fill the initial vector with random values

      // Create the eigenproblem.
      std::cout << "Setting eigen problem..." << std::endl;
      Teuchos::RCP<Anasazi::BasicEigenproblem<ScalarType, MV, OP> > MyProblem =
	Teuchos::rcp (new Anasazi::BasicEigenproblem<ScalarType, MV, OP> (A, ivec));

      // Set the number of eigenvalues requested
      MyProblem->setNEV (nev);

      // Inform the eigenproblem that you are finishing passing it information
      success = MyProblem->setProblem ();
      if (! success) {
	printer.print (Anasazi::Errors, "Anasazi::BasicEigenproblem::setProblem()\
 reported an error.\n");
	return EXIT_FAILURE;
      }

      // Create the solver manager
      std::cout << "Creating eigen solver..." << std::endl;
      Teuchos::RCP<Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP> > MySolverMan \
	= Teuchos::rcp (new Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP> (MyProblem, MyPL));
      
      // Solve the problem
      std::cout << "Solving eigen problem..." << std::endl;
      Anasazi::ReturnType returnCode = MySolverMan->solve ();

      // Get the eigenvalues and eigenvectors from the eigenproblem
      Anasazi::Eigensolution<ScalarType, MV> sol = MyProblem->getSolution ();
      std::vector<Anasazi::Value<ScalarType> > evals = sol.Evals;
      Teuchos::RCP<MV> evecs = sol.Evecs;
      std::vector<int> index = sol.index;
      int numev = sol.numVecs;
      std::cout << "Found " << numev << " eigenvalues" << std::endl;

      // Print eigenvalues and indices
      std::filebuf fb;
      std::ostream os(&fb);
      if (numev > 0)
	{
	  fb.open (EigValForwardFileName, std::ios::out);
	  for (int ev = 0; ev < numev; ev++)
	    os << evals[ev].realpart << "\t" << evals[ev].imagpart
	       << "\t" << index[ev] << std::endl;
	  fb.close();
	  
	  // Print eigenvectors
	  EpetraExt::MultiVectorToMatrixMarketFile(EigVecForwardFileName, *evecs);
	}

      // Solve adjoint problem
      if (stationary)
	{
	  std::cout << "Transposing transpose of forward transition matrix..." << std::endl;
	  Epetra_RowMatrixTransposer transposer(PT);
	  transposer.CreateTranspose(false, QT);
	}
      else
	{
	  std::cout << "Reading transpose of backward transition matrix..." << std::endl;
	  sprintf(backwardTransitionFileName,				\
		  "%s/transfer/backwardTransition/backwardTransition%s.coo", resDir, postfix);
	  std::cout << "reading at " << backwardTransitionFileName << std::endl;
	  EpetraExt::MatrixMarketFileToCrsMatrix(backwardTransitionFileName, Map, QT,
						 true);
	}
      A = Teuchos::rcpFromRef (*QT);
      
      ivec->Random (); // fill the initial vector with random values
      
      // Create the eigenproblem.
      std::cout << "Setting eigen problem..." << std::endl;
      MyProblem = Teuchos::rcp (new Anasazi::BasicEigenproblem<ScalarType, MV, OP> (A, ivec));

      // Set the number of eigenvalues requested
      MyProblem->setNEV (nev);

      // Inform the eigenproblem that you are finishing passing it information
      success = MyProblem->setProblem();
      if (! success) {
	printer.print (Anasazi::Errors, "Anasazi::BasicEigenproblem::setProblem()\
 reported an error.\n");
	return EXIT_FAILURE;
      }

      // Create the solver manager
      std::cout << "Creating eigen solver..." << std::endl;
      MySolverMan = Teuchos::rcp (new Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP>
				  (MyProblem, MyPL));
      
      // Solve the problem
      std::cout << "Solving eigen problem..." << std::endl;
      returnCode = MySolverMan->solve ();
      
      // Get the eigenvalues and eigenvectors from the eigenproblem
      sol = MyProblem->getSolution ();
      evals = sol.Evals;
      evecs = sol.Evecs;
      index = sol.index;
      numev = sol.numVecs;
      std::cout << "Found " << numev << " eigenvalues" << std::endl;

      if (stationary)
	{
	  for (int i = 0; i < N; i++)
	    {
	      if (gsl_vector_get(initDist, i) > 1.e-10)
		{
		  for (int ev = 0; ev < numev; ev++)
		    (*evecs)[ev][i] /= gsl_vector_get(initDist, i);
		}
	    }
	}

      // Print eigenvalues and indices
      if (numev > 0)
	{
	  // Print eigenvalues
	  fb.open (EigValBackwardFileName, std::ios::out);
	  for (size_t ev = 0; ev < numev; ev++)
	    os << evals[ev].realpart << "\t" << evals[ev].imagpart
	       << "\t" << index[ev] << std::endl;
	  fb.close();
      
	  // Print eigenvectors
	  EpetraExt::MultiVectorToMatrixMarketFile(EigVecBackwardFileName, *evecs);
	}


    }

  // Free
  freeConfig();
  
  return 0;
}
