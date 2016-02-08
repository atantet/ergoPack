#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <libconfig.h++>
#include <ergoPack/transferOperator.hpp>
#include <ergoPack/gsl_extension.h>

using namespace libconfig;


/** \file transfer.cpp
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


// Declarations
/** \brief User defined function to get parameters from a cfg file using libconfig. */
void readConfig(const char *cfgFileName);

// Configuration 
char resDir[256];               //!< Root directory in which results are written
char caseName[256];             //!< Name of the case to simulate 
char file_format[256];          //!< File format of output ("txt" or "bin")
int dim;                        //!< Dimension of the phase space
char delayName[256];            //!< Name associated with the number and values of the delays
double LCut;                    //!< Length of the time series without spinup
double spinup;                  //!< Length of initial spinup period to remove
double L;                       //!< Total length of integration
double dt;                      //!< Time step of integration
double printStep;               //!< Time step of output
size_t printStepNum;            //!< Time step of output in number of time steps of integration
size_t nt0;                     //!< Number of time steps of the source time series
size_t nt;                      //!< Number of time steps of the observable
int dimObs;                     //!< Dimension of the observable
size_t embedMax;                //!< Maximum lag for the embedding
gsl_vector_uint *components;    //!< Components in the time series used by the observable
gsl_vector_uint *embedding;     //!< Embedding lags for each component
gsl_vector_uint *nx;            //!< Number of grid boxes per dimension
gsl_vector *nSTDLow;            //!< Number of standard deviations below mean to span by the grid 
gsl_vector *nSTDHigh;           //!< Number of standard deviations above mean to span by the grid 
size_t nLags;                   //!< Number of transition lags for which to calculate the spectrum
gsl_vector *tauRng;             //!< Lags for which to calculate the spectrum
char srcPostfix[256];           //!< Postfix of simulation file.
char srcFileName[256];          //!< Name of the source simulation file
char obsName[256];              //!< Name associated with the observable
char gridPostfix[256];          //!< Postfix associated with the grid
char gridFileName[256];         //!< File name for the grid file
char configFileName[256];       //!< Name of the configuration file
bool stationary;                //!< Whether the problem is stationary or not


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
  char forwardTransitionFileName[256], initDistFileName[256], postfix[256];

  size_t tauNum;
  double tau;
  transferOperator *transferOp;


  // Get membership vector
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
  grid->printGrid(gridFileName, "%.12lf", stationary);
    
  // Open grid membership vector stream
  sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s.txt",
	  resDir, gridPostfix);
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

  
  // Get transition matrices for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);
      tauNum = (size_t) (tau / printStep);

      // Update file names
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));
      sprintf(forwardTransitionFileName,
	      "%s/transfer/forwardTransition/forwardTransition%s.coo", resDir, postfix);
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.txt", resDir, postfix);

      // Get full membership matrix
      std::cout << "Getting full membership matrix from the list of membership vecotrs..."
		<< std::endl;
      gridMemMatrix = memVector2memMatrix(gridMemVector, tauNum);

      // Get transition matrices as CSR
      std::cout << "Building stationary transfer operator..." << std::endl;
      transferOp = new transferOperator(gridMemMatrix, grid->getN(), true);
    
      // Write transition matrix as CSR
      std::cout << "Writing transfer operator..." << std::endl;
      transferOp->printForwardTransition(forwardTransitionFileName, "%.12lf");
      transferOp->printInitDist(initDistFileName, "%.12lf");
	
      // Free
      delete transferOp;
      gsl_matrix_uint_free(gridMemMatrix);
    }

  // Free
  gsl_vector_uint_free(components);
  gsl_vector_uint_free(embedding);
  delete grid;
  gsl_vector_uint_free(nx);
  gsl_vector_free(nSTDLow);
  gsl_vector_free(nSTDHigh);
  gsl_vector_free(tauRng);
  gsl_vector_uint_free(gridMemVector);
		
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
    
    // Dimension
    dim = cfg.lookup("model.dim");
    std::cout << "dim = " << dim << std::endl;
    
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

    // Output format
    strcpy(file_format, (const char *) cfg.lookup("simulation.file_format"));
    std::cout << "Output file format: " << file_format << std::endl;

    
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
    for (size_t d = 0; d < (size_t) (dimObs); d++)
      {
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

    stationary = cfg.lookup("transfer.stationary");
    std::cout << "Is stationary: " << stationary << std::endl;
      
    
    std::cout << std::endl;
    
    
    // Finish configuration
    // Define time series parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);
    nt0 = (size_t) (LCut * printStep);
    embedMax = gsl_vector_uint_max(embedding);
    nt = nt0 - embedMax;

    // Define postfix and src file name
    sprintf(srcPostfix, "_%s%s_L%d_spinup%d_dt%d_samp%d", caseName, delayName,
	    (int) L, (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)),
	    (int) printStepNum);
    sprintf(srcFileName, "%s/simulation/sim%s.%s", resDir, srcPostfix, file_format);

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
    sprintf(gridFileName, "%s/grid/grid%s.txt", resDir, gridPostfix);
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
