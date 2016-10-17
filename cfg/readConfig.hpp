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


using namespace libconfig;

// Configuration variables
extern char resDir[256];               //!< Root directory in which results are written
extern char caseName[256];             //!< Name of the case to simulate 
extern double rho;                     //!< Parameters for the Lorenz flow
extern double sigma;                   //!< Parameters for the Lorenz flow
extern double beta;                    //!< Parameters for the Lorenz flow
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
extern char srcPostfix[256];           //!< Postfix of simulation file.
extern char dstFileName[256];          //!< Destination file name
extern size_t nt0;                     //!< Number of time steps of the source time series
extern size_t nt;                      //!< Number of time steps of the observable
extern int dimObs;                     //!< Dimension of the observable
extern size_t embedMax;                //!< Maximum lag for the embedding
extern gsl_vector_uint *components;    //!< Components in the time series used by the observable
extern gsl_vector_uint *embedding;     //!< Embedding lags for each component
extern bool readGridMem;               //!< Whether to read the grid membership vector
extern size_t N;                       //!< Dimension of the grid
extern gsl_vector_uint *nx;            //!< Number of grid boxes per dimension
extern gsl_vector *nSTDLow;            //!< Number of standard deviations below mean to span by the grid 
extern gsl_vector *nSTDHigh;           //!< Number of standard deviations above mean to span by the grid 
extern size_t nLags;                   //!< Number of transition lags for which to calculate the spectrum
extern gsl_vector *tauRng;             //!< Lags for which to calculate the spectrum
extern int nev;                        //!< Number of eigenvectors to calculate
extern char obsName[256];              //!< Name associated with the observable
extern char gridPostfix[256];          //!< Postfix associated with the grid
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


/**
 * Sparse configuration file using libconfig++
 * to define all parameters of the case.
 */
void readConfig(const char *cfgFileName);

/**
 * Free memory allocated during configuration.
 */
void freeConfig();

#endif
