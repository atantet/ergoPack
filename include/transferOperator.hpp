#ifndef TRANSFEROPERATOR_HPP
#define TRANSFEROPERATOR_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ios>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <ergoPack/gsl_extension.h>
#include <ergoPack/ergoGrid.hpp>

/** \file transferOperator.hpp
 * \brief Calculate discretized approximation of transfer/Koopman operators from time series.
 *   
 * Calculate Galerkin approximation of transfer/Koopman operators from time series.
 * The result is given as forward and backward Markov transition matrices
 * approximating the transfer and Koopman operator, respectively,
 * as well as the initial and final distrtutions associated to them.

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
 * Transfer operator class including
 * the forward and backward transition matrices
 * and the initial and final distributions calculated from data.
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

  const size_t N;        //!< Size of the grid

  /** \brief Allocate memory. */
  int allocate();

  /** \brief Get the transition matrices from a grid membership matrix. */
  int buildFromMembership(const gsl_matrix_uint *gridMem);

  
public:
  gsl_spmatrix *P;       //!< Forward transition matrix (CRS)
  gsl_spmatrix *Q;       //!< Backward transition matrix (CRS)
  gsl_vector *rho0;      //!< Initial distribution
  gsl_vector *rhof;      //!< Final distribution

  
  /** \brief Empty constructor allocating for grid size*/
  transferOperator(size_t N_) : N(N_) { allocate(); }
  
  /** \brief Constructor from the membership matrix. */
  transferOperator(const gsl_matrix_uint *gridMem, size_t N_);
  
  /** \brief Constructor from initial and final states for a given grid */
  transferOperator(const gsl_matrix *initStates, const gsl_matrix *finalStates,
		   const Grid *grid);
  
  /** \brief Constructor from a long trajectory for a given grid and lag */
  transferOperator(const gsl_matrix *states, const Grid *grid, size_t tauStep);
  
  /** \brief Destructor */
  ~transferOperator();

  
  /** \brief Get number of grid boxes. */
  size_t getN() const { return N; }
  

  // Output methods
  /** \brief Print forward transition matrix to file in coordinate format.*/
  int printForwardTransition(const char *path, const char *dataFormat);
  
  /** \brief Print backward transition matrix to file in coordinate format.*/
  int printBackwardTransition(const char *path, const char *dataFormat);
  
  /** \brief Print initial distribution to file.*/
  int printInitDist(const char *path, const char *dataFormat);
  
  /** \brief Print final distribution to file.*/
  int printFinalDist(const char *path, const char *dataFormat);

  
  // Input methods
  /** \brief Scan forward transition matrix to file in coordinate format.*/
  int scanForwardTransition(const char *path);
  
  /** \brief Scan backward transition matrix to file in coordinate format.*/
  int scanBackwardTransition(const char *path);
  
  /** \brief Scan initial distribution to file.*/
  int scanInitDist(const char *path);
  
  /** \brief Scan final distribution to file.*/
  int scanFinalDist(const char *path);


  /** \brief Filtering of weak Markov states. */
  int filter(double tol);
};



/*
 *  Functions declarations
 */

/** \brief Get triplet vector from membership matrix. */
gsl_spmatrix *getTransitionCountTriplet(const gsl_matrix_uint *gridMem, size_t N);
/** \brief Remove weak nodes from a transition matrix. */
int filterStochasticMatrix(gsl_spmatrix *M, gsl_vector *rowCut, gsl_vector *colCut,
			   double tol, int norm);


/*
 * Constructors and destructors definitions
 */

/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem  GSL grid membership matrix.
 * \param[in] N_       Number of grid boxes.
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem, size_t N_)
  : N(N_)
{
  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);
}

/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the initial and final states of trajectories.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] grid           Pointer to Grid object.
 */
transferOperator::transferOperator(const gsl_matrix *initStates,
				   const gsl_matrix *finalStates,
				   const Grid *grid)
  : N(grid->getN())
{
  gsl_matrix_uint *gridMem;

  // Get grid membership matrix
  gridMem = getGridMemMatrix(initStates, finalStates, grid);

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);  
}

/**
 * Construct transferOperator calculating the forward and backward transition matrices
 * and distributions from a single long trajectory, for a given grid and lag.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] grid           Pointer to Grid object.
 * \param[in] tauStep        Lag used to calculate the transitions.
 */
transferOperator::transferOperator(const gsl_matrix *states, const Grid *grid,
				   const size_t tauStep)
  : N(grid->getN())
{
  gsl_matrix_uint *gridMem;

  // Get grid membership matrix from a single long trajectory
  gridMem = getGridMemMatrix(states, grid, tauStep);

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);
}

/** Destructor of transferOperator: desallocate all pointers. */
transferOperator::~transferOperator()
{
  gsl_spmatrix_free(P);
  gsl_spmatrix_free(Q);
  gsl_vector_free(rho0);
  gsl_vector_free(rhof);
}


/*
 * Methods definitions
 */

/**
 * Allocate memory for the transition matrices and distributions.
 * \param[in] N_ Number of grid boxes.
 * \return       Exit status.
 */
int
transferOperator::allocate()
{
  P = gsl_spmatrix_alloc(N, N);
  if (!P)
    {
      throw std::bad_alloc();
    }
  
  Q = gsl_spmatrix_alloc(N, N);
  if (!Q)
    {
      throw std::bad_alloc();
    }
  
  rho0 = gsl_vector_alloc(N);
  
  rhof = gsl_vector_alloc(N);
  
  return 0;
}

/**
 * Method called from the constructor to get the transition matrices
 * from a grid membership matrix.
 * \param[in] gridMem Grid membership matrix.
 * \return            Exit status.
 */
int
transferOperator::buildFromMembership(const gsl_matrix_uint *gridMem)
{
  gsl_spmatrix *T;

  // Get transition count triplets
  T = getTransitionCountTriplet(gridMem, N);
  
  /** Convert to CRS summing duplicates */
  if (!(P = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS)))
    {
      throw std::exception();
    }

  /** Get transpose copy */
  if (!(Q = gsl_spmatrix_alloc_nzmax(N, N, P->nz, GSL_SPMATRIX_CRS)))
    {
      throw std::bad_alloc();
    }
  if (gsl_spmatrix_transpose_memcpy(Q, P))
    {
      throw std::exception();
    }
  
  /** Get initial distribution */
  if (!(rho0 = gsl_spmatrix_get_colsum(Q)))
    {
      throw std::exception();
    }

  /** Get final distribution */
  if (!(rhof = gsl_spmatrix_get_colsum(P)))
    {
      throw std::exception();
    }
  
  /** Make left stochastic for matrix elements to be transition probabilities */
  gsl_spmatrix_div_cols(P, rhof);
  gsl_spmatrix_div_cols(Q, rho0);
  gsl_vector_normalize(rho0);
  gsl_vector_normalize(rhof);

  /** Free */
  gsl_spmatrix_free(T);

  return 0;
}


/** 
 * Filter weak Markov states (boxes) of forward and backward transition matrices
 * based on their distributions.
 * \param[in] tol Weight under which a state is filtered out.
 * \return        Exit status.
 */
int
transferOperator::filter(double tol)
{
  int status;
  
  /** Filter forward transition matrix */
  if (status = filterStochasticMatrix(P, rho0, rhof, tol, 2))
    {
      throw std::exception();
    }

  /** Filter forward transition matrix */
  if (status = filterStochasticMatrix(Q, rhof, rho0, tol, 2))
    {
      throw std::exception();
    }
  
  return 0;
}


/**
 * Print forward transition matrix to file in coordinate format with header
 * (see transferOperator.hpp)
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printForwardTransition(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printForwardTransition, \
opening stream to write");
    }

  // Print
  gsl_spmatrix_fprintf(fp, P, dataFormat);
  if (ferror(fp))
    {
      throw std::ios::failure("transferOperator::printForwardTransition, \
printing to transition matrix");
    }
  
  // Close
  fclose(fp);

  return 0;
}

/**
 * Print backward transition matrix to file in coordinate format with header
 * (see transferOperator.hpp)
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printBackwardTransition(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printBackwardTransition, \
opening stream to write");
    }

  // Print 
  gsl_spmatrix_fprintf(fp, Q, dataFormat);
  if (ferror(fp))
    {
      throw std::ios::failure("transferOperator::printBackwardTransition, \
printing transition matrix");
    }

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print initial distribution to file.
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printInitDist(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printInitDist, \
opening stream to write");
    }

  // Print
  gsl_vector_fprintf(fp, rho0, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print final distribution to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] dataFormat Format in which to print each element.
 * \return               Status.
 */
int
transferOperator::printFinalDist(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printFinalDist, \
opening stream to write");
    }

  // Print
  gsl_vector_fprintf(fp, rhof, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Scan forward transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to print.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanForwardTransition(const char *path)
{
  FILE *fp;
  gsl_spmatrix *T;
  
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanForwardTransition, \
opening stream to read");
  }

  /** Scan, summing if duplicate */
  if (!(T = gsl_spmatrix_fscanf(fp, 1)))
    {
      throw std::ios::failure("transferOperator::scanForwardTransition, \
scanning transition matrix");
    }

  /** Check if matrix dimension consistent with grid size */
  if ((T->size1 != T->size2) || (T->size1 != N))
    {
      throw std::length_error("transferOperator::scanForwardTransition, \
Triplet matrix size not consistent with this->N");
    }

  /** Compress */
  P = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS);
  gsl_spmatrix_free(T);
  if (!P)
    {
      throw std::exception();
    }

  //Close
  fclose(fp);
  
  return 0;
}

/**
 * Scan backward transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to print.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanBackwardTransition(const char *path)
{
  FILE *fp;
  gsl_spmatrix *T;
  
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanBackwardTransition, \
opening stream to read");
    }

  /** Scan, summing if duplicate */
  if (!(T = gsl_spmatrix_fscanf(fp, 1)))
    {
      throw std::ios::failure("transferOperator::scanBackwardTransition, \
scanning transition matrix");
  }

  /** Check if matrix dimension consistent with grid size */
  if ((T->size1 != T->size2) || (T->size1 != N))
    {
      throw std::length_error("transferOperator::scanForwardTransition, \
Triplet matrix size not consistent with this->N");
    }

  /** Compress */
  Q = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS);
  gsl_spmatrix_free(T);
  if (!Q)
    {
      throw std::exception();
    }

  //Close
  fclose(fp);
  
  return 0;
}


/**
 * Scan initial distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to print.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanInitDist(const char *path)
{
  FILE *fp;
    
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanInitDist, \
opening stream to read");
    }

  /** Scan after preallocating */
  rho0 = gsl_vector_alloc(N);
  gsl_vector_fscanf(fp, rho0);
  
  //Close
  fclose(fp);
  
  return 0;
}

/**
 * Scan final distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to print.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanFinalDist(const char *path)
{
  FILE *fp;
    
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanFinalDist, \
opening stream to read");
    }

  /** Scan after preallocating */
  rhof = gsl_vector_alloc(N);
  gsl_vector_fscanf(fp, rhof);
  
  //Close
  fclose(fp);
  
  return 0;
}


/*
 * Function definitions
 */

/** 
 * Get the triplet vector counting the transitions
 * from pairs of grid boxes from the grid membership matrix.
 * \param[in] gridMem Grid membership matrix.
 * \param[in] N       Size of the grid.
 * \return            Triplet vector counting the transitions.
 */
gsl_spmatrix *
getTransitionCountTriplet(const gsl_matrix_uint *gridMem, size_t N)
{
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  size_t nOut = 0;
  gsl_spmatrix *T;

  /** Allocate triplet sparse matrix */
  if (!(T = gsl_spmatrix_alloc_nzmax(N, N, nTraj, GSL_SPMATRIX_TRIPLET)))
    {
      std::bad_alloc();
    }

  /** Record transitions */
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    /** Add transition triplet, summing if duplicate */
    if ((box0 < N) && (boxf < N))
      gsl_spmatrix_set(T, box0, boxf, 1., 1);
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << "% of the trajectories ended up out of the domain." << std::endl;

  return T;
}


/**
 * Set weak states to 0 from a stochastic matrix and its distributions
 * based on initial and final probability of each Markov state.
 * Attention: elements are not actually removed from sparse matrix.
 * \param[inout] T    Compressed transition matrix to filter.
 * \param[in] rowCut  The probability distribution associated with each row.
 * \param[in] colCut  The probability distribution associated with each column.
 * \param[in] tol     Probability under which Markov states are removed.
 * \param[in] norm    Choice of normalization,
 * - norm = 0: normalize over all elements,
 * - norm = 1: to right stochastic,
 * - norm = 2: to left stochastic,
 * - no normalization for any other choice.
 */
int
filterStochasticMatrix(gsl_spmatrix *M,
		       gsl_vector *rowCut, gsl_vector *colCut,
		       double tol, int norm)
{
  size_t outerIdx, p, n;
  size_t isRowOut, isColOut;
  double totalSum;
  gsl_vector *sum;
  gsl_vector_uint *rowOut, *colOut;

  /** Check which nodes should be filtered
   * and set distribution to 0 */
  rowOut = gsl_vector_uint_calloc(M->size1);
  for (n = 0; n < M->size1; n++)
    {
      if (gsl_vector_get(rowCut, n) < tol)
	{
	  gsl_vector_uint_set(rowOut, n, 1);
	  gsl_vector_set(rowCut, n, 0.);
	}
    }
  
  colOut = gsl_vector_uint_calloc(M->size2);
  for (n = 0; n < M->size2; n++)
    {
      if (gsl_vector_get(colCut, n) < tol)
	{
	  gsl_vector_uint_set(colOut, n, 1);
	  gsl_vector_set(colCut, n, 0.);
	}
    }

  if (GSL_SPMATRIX_ISTRIPLET(M))
    {
      GSL_ERROR("first index out of range", GSL_EINVAL);
    }
  else if (GSL_SPMATRIX_ISCCS(M))
    {
      for (outerIdx = 0; outerIdx < M->outerSize; outerIdx++)
	{
	  isColOut = gsl_vector_uint_get(colOut, outerIdx);
	  for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; p++)
	    {
	      isRowOut = gsl_vector_uint_get(rowOut, M->i[p]);
	      // Remove elements of states to be removed
	      if (isRowOut || isColOut)
		M->data[p] = 0.;
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(M))
    {
      for (outerIdx = 0; outerIdx < M->outerSize; outerIdx++)
	{
	  isRowOut = gsl_vector_uint_get(rowOut, outerIdx);
	  for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; p++)
	    {
	      isColOut = gsl_vector_uint_get(colOut, M->i[p]);
	      // Remove elements of states to be removed
	      if (isRowOut || isColOut)
		M->data[p] = 0.;
	    }
	}
    }
    
  /** Make matrix and vectors stochastic again */
  switch (norm)
    {
    case 0:
      totalSum = gsl_spmatrix_get_sum(M);
      gsl_spmatrix_scale(M, 1. / totalSum);
      break;
    case 1:
      sum = gsl_spmatrix_get_rowsum(M);
      gsl_spmatrix_div_rows(M, sum);
      gsl_vector_free(sum);
      break;
    case 2:
      sum = gsl_spmatrix_get_colsum(M);
      gsl_spmatrix_div_cols(M, sum);
      gsl_vector_free(sum);
      break;
    default:
      break;
    }
  gsl_vector_normalize(rowCut);
  gsl_vector_normalize(colCut);

  /** Free */
  gsl_vector_uint_free(rowOut);
  gsl_vector_uint_free(colOut);

  return 0;
}


#endif
