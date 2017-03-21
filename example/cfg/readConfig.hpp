#ifndef READ_CONFIG_HPP
#define READ_CONFIG_HPP

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <libconfig.h++>
#include <configAR.hpp>
#include <ergoParam.hpp>


using namespace libconfig;

// Configuration variables
extern char resDir[256];               //!< Root directory in which results are written
extern char caseName[256];             //!< Name of the case to simulate 
extern param p;                        //!< Model adimensional parameters
extern char fileFormat[256];          //!< File format of output ("txt" or "bin")
extern char delayName[256];            //!< Name associated with the number and values of the delays
extern int dim;                        //!< Dimension of the phase space
extern gsl_vector *initState;          //!< Initial state for simulation
extern double LCut;                    //!< Length of the time series without spinup
extern double spinup;                  //!< Length of initial spinup period to remove
extern double L;                       //!< Total length of integration
extern double dt;                      //!< Time step of integration
extern double printStep;               //!< Time step of output
extern size_t printStepNum;            //!< Time step of output in number of time steps of integration
// Continuation
extern double epsDist;                 //!< Tracking distance tolerance
extern double epsStepCorrSize;         //!< Tracking correction step size tolerance
extern int maxIter;                    //!< Maximum number of iter. for corr.
extern int maxPred;                    //!< Maximum number of iter. for pred.
extern int numShoot;                   //!< Number of shoots
extern double contStep;                //!< Step size of parameter for continuation
extern double contMin;                 //!< Lower limit to which to continue
extern double contMax;                 //!< Limit to which to continue
extern bool verbose;                   //!< Verbose mode selection
extern gsl_vector *initCont;           //!< Initial state for continuation
extern char srcPostfix[256];           //!< Postfix of simulation file.
extern char dstFileName[256];          //!< Destination file name
// Sprinkle
extern int nTraj;                      //!< Number of trajectories to sprinkle
extern gsl_vector *minInitState;       //!< Lower limits of initial states of trajectories
extern gsl_vector *maxInitState;       //!< Upper limits of initial states of trajectories
extern gsl_vector_uint *seedRng;       //!< Seeds used to initialize the simulations
extern size_t nSeeds;                  //!< Number of seeds
extern char boxPostfix[256];           //!< Postfix associated with the bounding box
extern size_t nt0;                     //!< Number of time steps of the source time series
extern size_t nt;                      //!< Number of time steps of the observable
extern int dimObs;                     //!< Dimension of the observable
extern size_t embedMax;                //!< Maximum lag for the embedding
extern gsl_vector_uint *components;    //!< Components in the time series used by the observable
extern gsl_vector_uint *embedding;     //!< Embedding lags for each component
// Grid
extern char gridPostfix[256];          //!< Postfix associated with the grid
extern bool readGridMem;               //!< Whether to read the grid membership vector
extern size_t N;                       //!< Dimension of the grid
extern gsl_vector_uint *nx;            //!< Number of grid boxes per dimension
extern gsl_vector *gridLimitsLow;      //!< Grid limits
extern gsl_vector *gridLimitsUp;       //!< Grid limits
extern char gridLimitsType[32];        //!< Grid limits type
extern size_t nLags;                   //!< Number of transition lags for which to calculate the spectrum
extern gsl_vector *tauRng;             //!< Lags for which to calculate the spectrum
extern int nev;                        //!< Number of eigenvectors to calculate
extern char obsName[256];              //!< Name associated with the observable
extern char gridFileName[256];         //!< File name for the grid file
extern configAR config;                //!< Configuration data for the eigen problem
extern char configFileName[256];       //!< Name of the configuration file
extern bool stationary;                //!< Whether the problem is stationary or not
extern bool getForwardEigenvectors;    //!< Whether to get forward eigenvectors
extern bool getBackwardEigenvectors;   //!< Whether to get backward eigenvectors
extern bool makeBiorthonormal;         //!< Whether to make eigenvectors biorthonormal

/** \file readConfig.hpp
 *  \brief Routines to parse a configuration file with libconfig++
 */


/** \brief Sparse general configuration section. */
void readGeneral(const Config *cfg);

/** \brief Sparse model configuration section. */
void readModel(const Config *cfg);

/** \brief Sparse continuation configuration section. */
void readContinuation(const Config *cfg);

/** \brief Sparse simulation configuration section. */
void readSimulation(const Config *cfg);

/** \brief Sparse sprinkle configuration section. */
void readSprinkle(const Config *cfg);

/** \brief Sparse observable configuration section. */
void readObservable(const Config *cfg);

/** \brief Sparse grid configuration section. */
void readGrid(const Config *cfg);

/** \brief Sparse transfer configuration section. */
void readTransfer(const Config *cfg);

/** \brief Sparse spectrum configuration section. */
void readSpectrum(const Config *cfg);

/** \brief Sparse all configuration sections. */
void readConfig(const char *cfgFileName);

/** \brief Free memory allocated during configuration. */
void freeConfig();

#endif
