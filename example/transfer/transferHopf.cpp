#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <ergoGrid.hpp>
#include <transferOperator.hpp>
#include <gsl_extension.hpp>
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

  // Observable declarations
  char srcFileName[256];
  FILE *srcStream;
  std::vector<gsl_matrix *> statesTraj(nTraj);
  gsl_matrix *trajRaw;
  gsl_vector_view vIn, vOut;

  // Grid declarations
  Grid *grid;
  char gridFileName[256];

  // Grid membership declarations
  char gridMemFileName[256];
  FILE *gridMemStream;
  gsl_matrix_uint *gridMemMatrix;
  std::vector<gsl_vector_uint *> gridMemTraj(nTraj);
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    maskFileName[256], srcPostfix[256], dstPostfix[256], dstPostfixTau[256];

  size_t tauNum;
  double tau;
  transferOperator *transferOp;

  sprintf(srcPostfix, "_%s_mu%04d_alpha%04d_gamma%04d_delta%04d_beta%04d_eps%04d_L%d_spinup%d_dt%d_samp%d", caseName,
	  (int) (p["mu"] * 10000 + 0.1), (int) (p["alpha"] * 10000 + 0.1),
	  (int) (p["gamma"] * 10000 + 0.1), (int) (p["delta"] * 10000 + 0.1),
	  (int) (p["beta"] * 10000 + 0.1), (int) (p["eps"] * 10000 + 0.1),
	  (int) L, (int) spinup, (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  
  // Get grid membership matrix
  if (! readGridMem) {
    // Iterate one simulation per traj
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      // Get membership vector
      sprintf(srcFileName, "%s/simulation/sim%s_traj%d.%s",
	      resDir, srcPostfix, (int) traj, fileFormat);
	  
      // Open time series file
      if ((srcStream = fopen(srcFileName, "r")) == NULL) {
	fprintf(stderr, "Can't open source file %s for reading:",
		srcFileName);
	perror("");
	return(EXIT_FAILURE);
      }

      // Read one-dimensional time series
      std::cout << "Reading trajectory in " << srcFileName << std::endl;
      // Allocate trajectory and grid limits
      trajRaw = gsl_matrix_alloc(nt0, dim);
      if (strcmp(fileFormat, "bin") == 0)
	gsl_matrix_fread(srcStream, trajRaw);
      else
	gsl_matrix_fscanf(srcStream, trajRaw);

      // Get observable
      statesTraj[traj] = gsl_matrix_alloc(nt0, dimObs);
      for (size_t comp = 0; comp < (size_t) dimObs; comp++) {
	vIn = gsl_matrix_column(trajRaw,
				gsl_vector_uint_get(components, comp));
	vOut = gsl_matrix_column(statesTraj[traj], comp);
	gsl_vector_memcpy(&vOut.vector, &vIn.vector);
      }

      // Close trajectory file
      fclose(srcStream);

    }

    // Define grid
    grid = new RegularGrid(nx, gridLimitsLow, gridLimitsHigh);
    
    // Print grid
    sprintf(gridFileName, "%s/grid/grid_%s%s.txt", resDir, caseName,
	    gridPostfix);
    grid->printGrid(gridFileName, "%.12lf", true);
    sprintf(dstPostfix, "%s_nTraj%d%s", srcPostfix, (int) nTraj,
	    gridPostfix);

    
    // Get grid membership for each traj
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      // Grid membership file name
      sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s.%s",
	      resDir, dstPostfix, fileFormat);
  
      // Open grid membership vector stream
      if ((gridMemStream = fopen(gridMemFileName, "w")) == NULL) {
	fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	perror("");
	return(EXIT_FAILURE);
      }
    
      // Get grid membership vector
      std::cout << "Getting grid membership vector for traj "
		<< traj << std::endl;
      gridMemTraj[traj] = getGridMemVector(statesTraj[traj], grid);

      // Write grid membership
      if (strcmp(fileFormat, "bin") == 0)
	gsl_vector_uint_fwrite(gridMemStream, gridMemTraj[traj]);
      else
	gsl_vector_uint_fprintf(gridMemStream, gridMemTraj[traj], "%d");

      // Free states and close stream
      gsl_matrix_free(statesTraj[traj]);
      fclose(gridMemStream);
    }

    // Free
    delete grid;
  }
  else {
    // Read grid membership for each traj
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      // Grid membership file name
      sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s.%s",
	      resDir, dstPostfix, fileFormat);
	  
      // Open grid membership stream for reading
      std::cout << "Reading grid membership vector for traj "
		<< traj << " at " << gridMemFileName << std::endl;
	  
      if ((gridMemStream = fopen(gridMemFileName, "r")) == NULL) {
	fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	perror("");
	return(EXIT_FAILURE);
      }
      
      // Read grid membership
      gridMemTraj[traj] = gsl_vector_uint_alloc(nt);
      if (strcmp(fileFormat, "bin") == 0)
	gsl_vector_uint_fread(gridMemStream, gridMemTraj[traj]);
      else
	gsl_vector_uint_fscanf(gridMemStream, gridMemTraj[traj]);

      // Close stream
      fclose(gridMemStream);
    }
  }

  for (size_t itau = 0; itau < tauRng->size; itau++) {
    // Get transition matrices for different lags
    tau = gsl_vector_get(tauRng, itau);
    tauNum = (size_t) round(tau / printStep + 0.1);
    sprintf(dstPostfixTau, "%s_tau%04d", dstPostfix, (int) (tau * 10000 + 0.1));

    std::cout << "\nConstructing transfer operator for a lag of "
	      << tau << std::endl;


    // Get full membership matrix
    std::cout << "Getting full membership matrix from the list \
of membership vecotrs..." << std::endl;
    gridMemMatrix = memVectorList2memMatrix(&gridMemTraj, tauNum);

      
    // Get transition matrices as CSR
    std::cout << "Building stationary transfer operator..." << std::endl;
    transferOp = new transferOperator(gridMemMatrix, N);


    // Write results
    // Write forward transition matrix
    std::cout << "Writing forward transition matrix and initial distribution..."
	      << std::endl;
    sprintf(forwardTransitionFileName,
	    "%s/transfer/forwardTransition/forwardTransition%s.crs%s",
	    resDir, dstPostfixTau, fileFormat);
    transferOp->printTransition(forwardTransitionFileName,
				fileFormat, "%.12lf");

    // Write mask and initial distribution
    sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
	    resDir, dstPostfixTau, fileFormat);
    transferOp->printMask(maskFileName,
			  fileFormat, "%.12lf");

    sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
	    resDir, dstPostfixTau, fileFormat);
    transferOp->printInitDist(initDistFileName,
			      fileFormat, "%.12lf");
      
    // Free
    delete transferOp;
  }
  gsl_matrix_uint_free(gridMemMatrix);

  // Free
  for (size_t traj = 0; traj < (size_t) nTraj; traj++)
    gsl_vector_uint_free(gridMemTraj[traj]);
  freeConfig();
		
  return 0;
}
