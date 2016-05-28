#include "../cfg/readConfig.hpp"

// Configuration variables
char resDir[256];               //!< Root directory in which results are written
char caseName[256];             //!< Name of the case to simulate 
double rho;                     //!< Parameters for the Lorenz flow
double sigma;                   //!< Parameters for the Lorenz flow
double beta;                    //!< Parameters for the Lorenz flow
char file_format[256];          //!< File format of output ("txt" or "bin")
char delayName[256];            //!< Name associated with the number and values of the delays
int dim;                        //!< Dimension of the phase space
gsl_vector *initState;          //!< Initial state for simulation
double LCut;                    //!< Length of the time series without spinup
double spinup;                  //!< Length of initial spinup period to remove
double L;                       //!< Total length of integration
double dt;                      //!< Time step of integration
double printStep;               //!< Time step of output
size_t printStepNum;            //!< Time step of output in number of time steps of integration
char srcPostfix[256];           //!< Postfix of simulation file.
char srcFileName[256];          //!< Name of the source simulation file
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
/** Declare default structure looking for largest magnitude eigenvalues */
char configFileName[256];       //!< Name of the configuration file
bool stationary;                //!< Whether the problem is stationary or not
bool getForwardEigenvectors;    //!< Whether to get forward eigenvectors
bool getBackwardEigenvectors;   //!< Whether to get backward eigenvectors
bool makeBiorthonormal;         //!< Whether to make eigenvectors biorthonormal

/** \file readConfig.cpp
 *  \brief Definitions for readConfig.hpp
 */


void
readConfig(const char *cfgFileName)
{
  Config cfg;
  char cpyBuffer[256];
  configAR defaultCfgAR = {"LM", 0, 0., 0, NULL, true};
  
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

    // Dimension
    dim = cfg.lookup("model.dim");
    std::cout << "dim = " << dim << std::endl;
    
    // Case name
    strcpy(caseName, (const char *) cfg.lookup("model.caseName"));
    std::cout << "Case name: " << caseName << std::endl;
    if (cfg.exists("model.rho") & cfg.exists("model.sigma") & cfg.exists("model.beta"))
      {
	rho = cfg.lookup("model.rho");
	sigma = cfg.lookup("model.sigma");
	beta = cfg.lookup("model.beta");
	strcpy(cpyBuffer, caseName);
	sprintf(caseName, "%s_rho%d_sigma%d_beta%d", cpyBuffer,
		(int) (rho * 1000), (int) (sigma * 1000), (int) (beta * 1000));
      }	
    
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

    // Initial state
    if (cfg.exists("simulation.initState"))
      {
	const Setting &initStateSetting = cfg.lookup("simulation.initState");
	initState = gsl_vector_alloc(dim);
	std::cout << "initState = [";
	for (size_t i =0; i < (size_t) dim; i++)
	  {
	    gsl_vector_set(initState, i, initStateSetting[i]);
	    std::cout << gsl_vector_get(initState, i) << " ";
	  }
	std::cout << "]" << std::endl;
      }

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
	gsl_vector_uint_set(embedding, d,
			    (int) nearbyint(embd / 365 / printStep));
	std::cout << embd << " ";
	sprintf(cpyBuffer, obsName);
	sprintf(obsName, "%s_c%zu_e%d", cpyBuffer,
		gsl_vector_uint_get(components, d), (int) embd);
      }
    std::cout << "]" << std::endl;
			    
    
    // Get grid settings
    std::cout << std::endl << "---grid---" << std::endl;
    const Setting &nxSetting = cfg.lookup("grid.nx");
    nx = gsl_vector_uint_alloc(dimObs);
    N = 1;
    std::cout << "Number of grid boxes per dimension:" << std::endl;
    for (size_t d = 0; d < (size_t) (dimObs); d++)
      {
	gsl_vector_uint_set(nx, d, nxSetting[d]);
	N *= gsl_vector_uint_get(nx, d);
	std::cout << "dim " << d+1 << ": "
		  << gsl_vector_uint_get(nx, d) << std::endl;
    }
    if (cfg.exists("grid.nSTDLow") & cfg.exists("grid.nSTDLow"))
      {
	const Setting &nSTDLowSetting = cfg.lookup("grid.nSTDLow");
	const Setting &nSTDHighSetting = cfg.lookup("grid.nSTDHigh");
	nSTDLow = gsl_vector_alloc(dimObs);
	nSTDHigh = gsl_vector_alloc(dimObs);
	std::cout << "Grid limits (nSTDLow, nSTDHigh):" << std::endl;
	for (size_t d = 0; d < (size_t) (dimObs); d++)
	  {
	    gsl_vector_set(nSTDLow, d, nSTDLowSetting[d]);
	    gsl_vector_set(nSTDHigh, d, nSTDHighSetting[d]);
	    std::cout << "dim " << d+1 << ": ("
		      << gsl_vector_get(nSTDLow, d) << ", "
		      << gsl_vector_get(nSTDHigh, d) << ")" << std::endl;
	  }
      }
    else
      {
	nSTDLow = NULL;
	nSTDHigh = NULL;
      }
    if (cfg.exists("grid.readGridMem"))
      readGridMem = cfg.lookup("grid.readGridMem");
    else
      readGridMem = false;
    std::cout << "readGridMem: " << readGridMem << std::endl;


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

    if (cfg.exists("spectrum.getForwardEigenvectors"))
      {
	getForwardEigenvectors = cfg.lookup("spectrum.getForwardEigenvectors");
      }
    if (cfg.exists("spectrum.getBackwardEigenvectors"))
      {
	getBackwardEigenvectors = cfg.lookup("spectrum.getBackwardEigenvectors");
      }
    if (cfg.exists("spectrum.makeBiorthonormal"))
      {
	makeBiorthonormal = cfg.lookup("spectrum.makeBiorthonormal");
      }

    // Finish configuration
    // Define time series parameters
    L = LCut + spinup;
    printStepNum = (size_t) (printStep / dt);
    nt0 = (size_t) (LCut / printStep);
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
      if (nSTDLow && nSTDHigh)
	{
	  sprintf(gridPostfix, "%s_n%dl%dh%d", cpyBuffer,
		  gsl_vector_uint_get(nx, d),
		  (int) gsl_vector_get(nSTDLow, d),
		  (int) gsl_vector_get(nSTDHigh, d));
	}
      else
	{
	  sprintf(gridPostfix, "%s_n%dminmax", cpyBuffer,
		  gsl_vector_uint_get(nx, d));
	}
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


/**
 * Free memory allocated during configuration.
 */
void
freeConfig()
{
  gsl_vector_free(tauRng);
  gsl_vector_free(nSTDHigh);
  gsl_vector_free(nSTDLow);
  gsl_vector_uint_free(nx);
  gsl_vector_uint_free(embedding);
  gsl_vector_uint_free(components);

  return;
}
