#ifndef CONFIGAR_HPP
#define CONFIGAR_HPP

/** \addtogroup transfer
 * @{
 */

/** \file configAR.hpp
 *  \brief Configuration structures and parameters for ARPACK++
 *   
 *  Configuration structures and parameters for ARPACK++
 */


/**
 * \brief Utility structure used to give configuration options to ARPACK++.
 * 
 * Utility structure used to give configuration options to ARPACK++.
 */
typedef struct {
  char *which; //!< Which eigenvalues to look for. 'LM' for Largest Magnitude
  int ncv;           //!< The number of Arnoldi vectors generated at each iteration of ARPACK
  double tol;        //!< The relative accuracy to which eigenvalues are to be determined
  int maxit;         //!< The maximum number of iterations allowed
  double *resid;     //!< A starting vector for the Arnoldi process
  bool AutoShift;    /**< Use ARPACK++ generated exact shifts for the implicit restarting
		      *   of the Arnoldi or one supplied by user */
} configAR;

/**
 * @}
 */

#endif
