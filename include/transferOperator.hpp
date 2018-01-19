#ifndef TRANSFEROPERATOR_HPP
#define TRANSFEROPERATOR_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ios>
#include <limits>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl_extension.hpp>
#include <ergoGrid.hpp>


/** \addtogroup transfer
 * @{
 */

/** \file transferOperator.hpp
 * \brief Calculate discretized approximation of transfer/Koopman operators from time series.
 *   
 * Calculate Galerkin approximation of transfer/Koopman operators from time series.
 * The result is given as a Markov transition matrix
 * approximating the transfer and Koopman operator, respectively,
 * as well as the initial distrtution associated to them.

 * Coordinate format with a one line header is followed
 * for printing and scanning of the sparse transition matrices.
 * The first line is a header giving the number of rows, columns and non-zero elements:
 *      M    N    nnz
 * Each following line gives the coordinate of one matrix element
 * as its row, column and value:
 *      i    j    d
 */


/*
 * Class declarations
 */

/** \brief Transfer operator class.
 * 
 * Transfer operator class including the transition matrix
 * and the initial distribution calculated from data.
 * The constructors are based on membership matrices 
 * with the first column giving the box to which belong the initial state of trajectories
 * and the second column the box to which belong the final state of trajectories.
 * The initial and final states can also be directly given as to matrices
 * with each row giving a state.
 * Finally, transferOperator can also be constructed 
 * from a single long trajectory and a given lag.
 * Then, a membership vector is first calculated,
 * assigning each realization to a box,
 * from which the membership matrix can be calculated
 * for the lag given.
 * 
 * The transition matrices are in CRS format,
 * allowing for instant transpose to CCS
 * for eigen problem using ARPACK++.
 */
class transferOperator {

  const size_t N;  //!< Size of the grid
  size_t NFilled;  //!< Number of filled boxes

  /** \brief Get the transition matrices from a grid membership matrix. */
  int buildFromMembership(const gsl_matrix_uint *gridMem);

  
public:
  gsl_spmatrix *P;       //!< Transition matrix (CRS)
  gsl_vector *initDist;  //!< Initial distribution
  gsl_vector_uint *mask; //!< Mask for empty boxes

  
/** \brief Empty constructor to allow manual building of transition matrix. */
  transferOperator(const size_t N_);
  
  /** \brief Constructor from the membership matrix. */
  transferOperator(const gsl_matrix_uint *gridMem, const size_t N_);
  
  /** \brief Constructor from initial and final states for a given grid */
  transferOperator(const gsl_matrix *initStates,
		   const gsl_matrix *finalStates, const Grid *grid);
  
  /** \brief Constructor from a long trajectory for a given grid and lag */
  transferOperator(const gsl_matrix *states, const Grid *grid,
		   size_t tauStep);
  
  /** \brief Destructor */
  ~transferOperator();

  /** \brief Allocate memory to distribution. */
  int allocateDist();

  
  /** \brief Get number of grid boxes. */
  size_t getN() const { return N; }

  /** \brief Get number of filled grid boxes. */
  size_t getNFilled() const { return NFilled; }


  // Output methods
  /** \brief Print transition matrix to file in coordinate format.*/
  int printTransition(const char *path,
		      const char *fileFormat, const char *dataFormat);
  
  /** \brief Print initial distribution to file.*/
  int printInitDist(const char *path,
		    const char *fileFormat, const char *dataFormat);
  
  /** \brief Print mask to file.*/
  int printMask(const char *path,
		const char *fileFormat, const char *dataFormat);

  
  // Input methods
  /** \brief Scan transition matrix to file in coordinate format.*/
  int scanTransition(const char *path, const char *fileFormat);
  
  /** \brief Scan initial distribution from file.*/
  int scanInitDist(const char *path,
		   const char *fileFormat);
  
  /** \brief Scan mask from file.*/
  int scanMask(const char *path,
	       const char *fileFormat);


  // Manual modifications
  /** \brief Manually change the initial distribution. */
  void setInitDist(const gsl_vector *initDist_)
  { gsl_vector_memcpy(initDist, initDist_); return; }

  /** \brief Manually change the mask. */
  void setMask(const gsl_vector_uint *mask_)
  { gsl_vector_uint_memcpy(mask, mask_); return; }

  /** \brief Add transition from pairs of initial and final grid boxes. */
  size_t addTransition(gsl_spmatrix *T, size_t box0, size_t boxf);

  /** \brief Get triplet vector from membership matrix. */
  size_t getTransitionCountTriplet(const gsl_matrix_uint *gridMem,
				   gsl_spmatrix *T);

  /** \brief Buid transition matrices from transition count matrix. */
  void buildFromTransitionCount(const gsl_spmatrix *T, const size_t nTraj);
    
  /** \brief Filtering of weak Markov states. */
  int filter(double tol);
};



/*
 *  Functions declarations
 */

/** \brief Get mask from grid membership matrix. */
size_t getMask(gsl_vector_uint *mask, const gsl_matrix_uint *gridMem=NULL);

/** \brief Remove weak nodes from a transition matrix. */
int filterStochasticMatrix(gsl_spmatrix *M,
			   gsl_vector *rowCut, gsl_vector *colCut,
			   double tol, int norm);


/**
 * @}
 */

#endif
