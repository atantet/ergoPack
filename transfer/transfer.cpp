#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <ergoPack/transferOperator.hpp>
#include <ergoPack/gsl_extension.h>
#include "../cfg/readConfig.hpp"


/** \file transfer.cpp
 *  \ingroup examples
 *  \brief Get transition matrices and distributions directly from time series.
 *   
 * Get transition matrices and distributions from a long time series
 * (e.g. simulation output).
 * Takes as first a configuration file to be parsed with libconfig C++ library.
 * First read the observable and get its mean and standard deviation
 * used to adapt the grid.
 * A rectangular grid is used here.
 * A grid membership vector is calculated for each time series 
 * assigning to each realization a grid box.
 * Then, the membership matrix is calculated for a given lag.
 * The forward transition matrices as well as the initial distributions
 * are calculated from the membership matrix.
 * Note that, since the transitions are calculated from long time series,
 * the problem must be autonomous and ergodic (stationary) so that
 * the backward transition matrix and final distribution need not be calculated.
 * Finally, the results are printed.
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


/** \brief Calculate transfer operators from time series.
 *
 *  After parsing the configuration file,
 *  the time series is read and an observable is designed
 *  selecting components with a given embedding lag.
 *  A membership vector is then built from the observable,
 *  attaching the box they belong to to every realization.
 *  The membership vector is then converted to a membership
 *  matrix for different lags and the transfer operators 
 *  built. The results are then written to file.
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

  // Observable declarations
  FILE *srcStream;
  gsl_matrix *traj;
  gsl_matrix *states;

  // Grid declarations
  Grid *grid;

  // Grid membership declarations
  char gridMemFileName[256];
  FILE *gridMemStream;
  gsl_vector_uint *gridMemVector;
  gsl_matrix_uint *gridMemMatrix;
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    postfix[256];

  size_t tauNum;
  double tau;
  transferOperator *transferOp;


  // Get membership vector
  sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s.txt",
	  resDir, gridPostfix);
  if (! readGridMem)
    {
      // Open time series file
      if ((srcStream = fopen(srcFileName, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open source file %s for reading:", srcFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}

      // Read one-dimensional time series
      std::cout << "Reading trajectory in " << srcFileName << std::endl;
      traj = gsl_matrix_alloc(nt0, dim);
      gsl_matrix_fread(srcStream, traj);

      // Close 
      fclose(srcStream);


      // Define observable
      states = gsl_matrix_alloc(nt, (size_t) dimObs);
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  gsl_vector_const_view view 
	    = gsl_matrix_const_subcolumn(traj, gsl_vector_uint_get(components, d),
					 embedMax - gsl_vector_uint_get(embedding, d),
					 nt);
	  gsl_matrix_set_col(states, d, &view.vector);
	}


      // Define grid
      grid = new RegularGrid(nx, nSTDLow, nSTDHigh, states);
    
      // Print grid
      grid->printGrid(gridFileName, "%.12lf", true);
    
      // Open grid membership vector stream
      if ((gridMemStream = fopen(gridMemFileName, "w")) == NULL){
	fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	perror("");
	return(EXIT_FAILURE);
      }
    
      // Get grid membership vector
      std::cout << "Getting grid membership vector" << std::endl;
      gridMemVector = getGridMemVector(states, grid);

      // Write grid membership
      gsl_vector_uint_fprintf(gridMemStream, gridMemVector, "%d");
    
      // Close stream and free
      fclose(gridMemStream);
      gsl_matrix_free(traj);
      gsl_matrix_free(states);
      delete grid;
    }
  else
    {
      // Open grid membership stream for reading
      std::cout << "Reading grid membership vector at " << gridMemFileName << std::endl;
      if ((gridMemStream = fopen(gridMemFileName, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}
      
      // Read grid membership
      gridMemVector = gsl_vector_uint_alloc(nt);
      gsl_vector_uint_fscanf(gridMemStream, gridMemVector);

      // Close stream
      fclose(gridMemStream);
    }



  // Get transition matrices for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);
      tauNum = (size_t) (tau / printStep);
      std::cout << "\nConstructing transfer operator for a lag of " << tau << std::endl;

      // Update file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));
      sprintf(forwardTransitionFileName,
	      "%s/transfer/forwardTransition/forwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt", resDir, postfix);
      sprintf(backwardTransitionFileName,
	      "%s/transfer/backwardTransition/backwardTransition%s.coo", resDir, postfix);
      sprintf(finalDistFileName, "%s/transfer/finalDist/finalDist%s.txt", resDir, postfix);

      
      // Get full membership matrix
      std::cout << "Getting full membership matrix from the list of membership vecotrs..."
		<< std::endl;
      gridMemMatrix = memVector2memMatrix(gridMemVector, tauNum);

      
      // Get transition matrices as CSR
      std::cout << "Building stationary transfer operator..." << std::endl;
      transferOp = new transferOperator(gridMemMatrix, N, stationary);
      
      // Write forward transition matrix
      std::cout << "Writing forward transition matrix and initial distribution..." << std::endl;
      transferOp->printForwardTransition(forwardTransitionFileName, "%.12lf");
      transferOp->printInitDist(initDistFileName, "%.12lf");
      
      // Write backward transition matrix
      if (!stationary)
	{
	  std::cout << "Writing backward transition matrix and final distribution..." << std::endl;
	  transferOp->printBackwardTransition(backwardTransitionFileName, "%.12lf");
	  transferOp->printFinalDist(finalDistFileName, "%.12lf");
	}
	
      // Free
      delete transferOp;
      gsl_matrix_uint_free(gridMemMatrix);
    }

  // Free
  gsl_vector_uint_free(gridMemVector);
  freeConfig();
		
  return 0;
}
