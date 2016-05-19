#include <transferOperator.hpp>

/** \file transferOperator.cpp
 *  \brief Definitions for transferOperator.hpp
 *
 */

/*
 * Constructors and destructors definitions
 */

/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem     GSL grid membership matrix.
 * \param[in] N_          Number of grid boxes.
 * \param[in] stationary_ Whether the system is stationary
 *                        (in which case \f$\rho_0 = \rho_f\f$ and no need
 *                        to calculate the backward transition matrix).
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem, const size_t N_,
				   const bool stationary_=false)
  : N(N_), stationary(stationary_)
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
 * \param[in] stationary_    Whether the system is stationary
 *                           (in which case \f$\rho_0 = \rho_f\f$ and no need
 *                           to calculate the backward transition matrix).
 */
transferOperator::transferOperator(const gsl_matrix *initStates,
				   const gsl_matrix *finalStates,
				   const Grid *grid, const bool stationary_=false)
  : N(grid->getN()), stationary(stationary_)
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
  : N(grid->getN()), stationary(true)
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
  if (P)
    gsl_spmatrix_free(P);
  gsl_vector_free(initDist);

  if (Q)
    gsl_spmatrix_free(Q);
  if (!stationary)
    {
      gsl_vector_free(finalDist);
    }
}


/*
 * Methods definitions
 */

/**
 * Allocate memory for the transition matrices and distributions.
 * \return               Exit status.
 */
int
transferOperator::allocate()
{
  P = NULL;
  initDist = gsl_vector_alloc(N);
  
  /** If stationary problem, initDist = finalDist = stationary distribution
   *  and the backward transition matrix need not be calculated */
  Q = NULL;
  if (!stationary)
    {    
      finalDist = gsl_vector_alloc(N);
    }
  else
    {
      finalDist = initDist;
    }
  
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
  const size_t nTraj = gridMem->size1;
  gsl_spmatrix *T;
  const double tol = std::numeric_limits<double>::epsilon() * N * 100;

  // Allocate distributions and set matrices to NULL pointer
  allocate();


  // Get transition count triplets
  if (!(T = gsl_spmatrix_alloc_nzmax(N, N, nTraj, GSL_SPMATRIX_TRIPLET)))
    {
      fprintf(stderr, "getTransitionCountTriplet: error allocating\
triplet count matrix.\n");
      std::bad_alloc();
    }

  if (stationary)
    getTransitionCountTriplet(gridMem, N, T, initDist, NULL);
  else
    getTransitionCountTriplet(gridMem, N, T, initDist, finalDist);

  
  /** Convert to CRS summing duplicates */
  if (!(P = gsl_spmatrix_crs(T)))
    {
      fprintf(stderr, "transferOperator::buildFromMembership: error compressing\
forward transition matrix.\n");
      throw std::exception();
    }

  /** If the problem is not stationary, get final distribution
   *  and backward transition matrix */
  if (!stationary)
    {
      /** Get transpose copy */
      if (!(Q = gsl_spmatrix_alloc_nzmax(N, N, P->nz, GSL_SPMATRIX_CRS)))
	{
	  fprintf(stderr, "transferOperator::buildFromMembership: error allocating\
backward transition matrix.\n");
	  throw std::bad_alloc();
	}
      if (gsl_spmatrix_transpose_memcpy(Q, P))
	{
	  fprintf(stderr, "transferOperator::buildFromMembership: error copying\
forward transition matrix to backward.\n");
	  throw std::exception();
	}
  
      /** Make the backward transition matrix left stochastic */
      gsl_spmatrix_div_cols(Q, initDist, tol);
      gsl_vector_normalize(initDist);
    }
  
  /** Make the forward transition matrix left stochastic */
  gsl_spmatrix_div_cols(P, finalDist, tol);
  gsl_vector_normalize(finalDist);
  
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
  if ((status = filterStochasticMatrix(P, initDist, finalDist, tol, 2)))
    {
      fprintf(stderr, "transferOperator::filter: error filtering\
forward transition matrix.\n");
      throw std::exception();
    }

  /** Filter forward transition matrix */
  if (Q)
    {
      if ((status = filterStochasticMatrix(Q, finalDist, initDist, tol, 2)))
	{
	  fprintf(stderr, "transferOperator::filter: error filtering\
backward transition matrix.\n");
	  throw std::exception();
	}
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

  if (Q)
    {
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
    }
  else
    {
      throw std::ios::failure("transferOperator::printBackwardTransition, \
backward transition matrix not calculated.");
    }

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
  gsl_vector_fprintf(fp, initDist, dataFormat);

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
  gsl_vector_fprintf(fp, finalDist, dataFormat);
  
  // Close
  fclose(fp);

  return 0;
}

/**
 * Scan forward transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to scan.
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
  if (!(T = gsl_spmatrix_fscanf(fp)))
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
  P = gsl_spmatrix_crs(T);
  gsl_spmatrix_free(T);
  if (!P)
    {
      fprintf(stderr, "transferOperator::scanForwardTransition: error\
compressing forward transition matrix.\n");
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
 * \param[in] path Path to the file in which to scan.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanBackwardTransition(const char *path)
{
  FILE *fp;
  gsl_spmatrix *T;
  
  // Open file
  if (!stationary)
    {
      if (!(fp = fopen(path, "r")))
	{
	  throw std::ios::failure("transferOperator::scanBackwardTransition, \
opening stream to read");
	}

      /** Scan, summing if duplicate */
      if (!(T = gsl_spmatrix_fscanf(fp)))
	{
	  throw std::ios::failure("transferOperator::scanBackwardTransition, \
scanning transition matrix");
	}

      /** Check if matrix dimension consistent with grid size */
      if ((T->size1 != T->size2) || (T->size1 != N))
	{
	  throw std::length_error("transferOperator::scanBackwardTransition, \
Triplet matrix size not consistent with this->N");
	}

      /** Compress */
      Q = gsl_spmatrix_crs(T);
      gsl_spmatrix_free(T);
      if (!Q)
	{
	  fprintf(stderr, "transferOperator::scanBackwardTransition: error\
compressing backward transition matrix.\n");
	  throw std::exception();
	}

      //Close
      fclose(fp);
    }
  else
    {
      throw std::ios::failure("transferOperator::scanBackwardTransition:\
backward transition matrix not scanned because problem is stationary");
    }

  
  return 0;
}


/**
 * Scan initial distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to scan.
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
  gsl_vector_fscanf(fp, initDist);
  
  //Close
  fclose(fp);
  
  return 0;
}

/**
 * Scan final distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path Path to the file in which to scan.
 * \return         0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanFinalDist(const char *path)
{
  FILE *fp;

  if (!stationary)
    {
      // Open file
      if (!(fp = fopen(path, "r")))
	{
	  throw std::ios::failure("transferOperator::scanFinalDist, \
opening stream to read");
	}

      /** Scan after preallocating */
      gsl_vector_fscanf(fp, finalDist);
  
      //Close
      fclose(fp);
    }
  else
    {
      throw std::ios::failure("transferOperator::scanFinalDist, \
final distribution not scanned because problem is stationary");
    }

  return 0;
}


/*
 * Function definitions
 */

/** 
 * Get the triplet vector counting the transitions
 * from pairs of grid boxes from the grid membership matrix.
 * \param[in]  gridMem Grid membership matrix.
 * \param[in]  N       Size of the grid.
 * \param[out] T       Triplet matrix counting the transitions.
 * \param[out] initDist    Initial states count
 * \param[out] finalDist    Final states count
 */
void
getTransitionCountTriplet(const gsl_matrix_uint *gridMem, size_t N,
			  gsl_spmatrix *T, gsl_vector *initDist=NULL, gsl_vector *finalDist=NULL)
{
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  size_t nOut = 0;
  double *ptr;

  /** Record transitions */
  if (initDist)
    gsl_vector_set_zero(initDist);
  if (finalDist)
    gsl_vector_set_zero(finalDist);
  for (size_t traj = 0; traj < nTraj; traj++)
    {
      box0 = gsl_matrix_uint_get(gridMem, traj, 0);
      boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
      /** Add transition triplet, summing if duplicate */
      if ((box0 < N) && (boxf < N))
	{
	  ptr = gsl_spmatrix_ptr(T, box0, boxf);
	  if (ptr)
	    *ptr += 1.; /* sum duplicate values */
	  else
	    gsl_spmatrix_set(T, box0, boxf, 1.);   /* initalize to x */

	  // Add initial and final boxes to initial and final distributions, respectively
	  if (initDist)
	    gsl_vector_set(initDist, box0, gsl_vector_get(initDist, box0) + 1.);
	  if (finalDist)
	    gsl_vector_set(finalDist, boxf, gsl_vector_get(finalDist, boxf) + 1.);
	}
      else
	nOut++;
    }
  std::cout <<  nOut * 100. / nTraj
	    << "% of the trajectories ended up out of the domain." << std::endl;

  return;
}


/**
 * Set weak states to 0 from a stochastic matrix and its distributions
 * based on initial and final probability of each Markov state.
 * Attention: elements are not actually removed from sparse matrix.
 * \param[in,out] M       Compressed transition matrix to filter.
 * \param[in]     rowCut  The probability distribution associated with each row.
 * \param[in]     colCut  The probability distribution associated with each column.
 * \param[in]     tol     Probability under which Markov states are removed.
 * \param[in]     norm    Choice of normalization,
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
  double eps;

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
      for (outerIdx = 0; outerIdx < M->size2; outerIdx++)
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
      for (outerIdx = 0; outerIdx < M->size1; outerIdx++)
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
      eps = std::numeric_limits<double>::epsilon() * M->size2 * 100;
      sum = gsl_vector_alloc(M->size1);
      gsl_spmatrix_get_rowsum(sum, M);
      gsl_spmatrix_div_rows(M, sum, eps);
      gsl_vector_free(sum);
      break;
    case 2:
      eps = std::numeric_limits<double>::epsilon() * M->size1 * 100;
      sum = gsl_vector_alloc(M->size1);
      gsl_spmatrix_get_colsum(sum, M);
      gsl_spmatrix_div_cols(M, sum, eps);
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


