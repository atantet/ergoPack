#include <transferOperator.hpp>
#include <cstring>

/** \file transferOperator.cpp
 *  \brief Definitions for transferOperator.hpp
 *
 */

/*
 * Constructors and destructors definitions
 */

/**
 * Empty constructor to allow manual building of transition matrix
 * from the grid membership matrix without mask.
 * \param[in] N_          Number of grid boxes.
 *                        (in which case \f$\rho_0 = \rho_f\f$).
 */
transferOperator::transferOperator(const size_t N_): N(N_), P(NULL)
{
  // Get default mask
  mask = gsl_vector_uint_alloc(N);
  NFilled = getMask(mask);

  // Allocate distributions and set matrices to NULL pointer
  allocateDist();

  /** Initialize distributions to zero */
  gsl_vector_set_zero(initDist);
}

/**
 * Construct transferOperator by calculating
 * the transition matrix and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem     GSL grid membership matrix.
 * \param[in] N_          Number of grid boxes.
 *                        (in which case \f$\rho_0 = \rho_f\f$).
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem,
				   const size_t N_) : N(N_), P(NULL)
{
  // Get mask
  mask = gsl_vector_uint_alloc(N);
  NFilled = getMask(mask, gridMem);

  // Allocate distributions and set matrices to NULL pointer
  allocateDist();
  
  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);
}

/**
 * Construct transferOperator by calculating
 * the transition matrix and distributions 
 * from the initial and final states of trajectories.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates     GSL matrix of final states.
 * \param[in] grid           Pointer to Grid object.
 *                           (in which case \f$\rho_0 = \rho_f\f$).
 */
transferOperator::transferOperator(const gsl_matrix *initStates,
				   const gsl_matrix *finalStates,
				   const Grid *grid)
  : N(grid->getN()), P(NULL)
{
  gsl_matrix_uint *gridMem;

  // Get grid membership matrix
  gridMem = getGridMemMatrix(initStates, finalStates, grid);

  // Get mask
  mask = gsl_vector_uint_alloc(N);
  NFilled = getMask(mask, gridMem);

  // Allocate distribution
  allocateDist();

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);  
}

/**
 * Construct transferOperator calculating the transition matrix
 * and distributions from a single long trajectory, for a given grid and lag.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] grid           Pointer to Grid object.
 * \param[in] tauStep        Lag used to calculate the transitions.
 */
transferOperator::transferOperator(const gsl_matrix *states, const Grid *grid,
				   const size_t tauStep)
  : N(grid->getN()), P(NULL)
{
  gsl_matrix_uint *gridMem;

  // Get grid membership matrix from a single long trajectory
  gridMem = getGridMemMatrix(states, grid, tauStep);

  // Get mask
  mask = gsl_vector_uint_alloc(N);  
  NFilled = getMask(mask, gridMem);

  // Allocate distributions
  allocateDist();

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
  gsl_vector_uint_free(mask);
}


/*
 * Methods definitions
 */

/**
 * Allocate memory for the distributions.
 * \return Exit status.
 */
int
transferOperator::allocateDist()
{
  initDist = gsl_vector_alloc(NFilled);
  
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
  size_t nIn;
  gsl_spmatrix *T;

  // Allocate transition count matrix
  if (!(T = gsl_spmatrix_alloc_nzmax(NFilled, NFilled, nTraj,
				     GSL_SPMATRIX_TRIPLET)))
    {
      fprintf(stderr, "buildFromMembership: error allocating\
triplet count matrix.\n");
      std::bad_alloc();
    }

  // Get transitions
  nIn = getTransitionCountTriplet(gridMem, T);
  std::cout <<  nIn * 100. / nTraj
	    << "% of the trajectories ended up inside the domain." << std::endl;

  // Build transition matrices from transition count matrix
  buildFromTransitionCount(T, nTraj);
  
  /** Free transition count matrix */
  gsl_spmatrix_free(T);

  return 0;
}


/**
 * Buid transition matrices of transfer operator from transition count matrix.
 * \param[in]  T    Transition count matrix.
 * \param[in] nTraj Number of trajectories (to define the minimum tolerance).
 */
void
transferOperator::buildFromTransitionCount(const gsl_spmatrix *T,
					   const size_t nTraj)
{
  const double tol = .9 / nTraj; // The min non-zero dist is 1. / nTraj
  
  /** Convert to CRS summing duplicates */
  if (!(P = gsl_spmatrix_crs(T)))
    {
      fprintf(stderr, "transferOperator::buildFromTransitionCount: \
error compressing transition matrix.\n");
      throw std::exception();
    }

  /** Make the transition matrix right stochastic */
  gsl_spmatrix_div_rows(P, initDist, tol);
  gsl_vector_normalize(initDist);

  return;
}

/** 
 * Filter weak Markov states (boxes) of the transition matrix
 * based on their distributions.
 * \param[in] tol Weight under which a state is filtered out.
 * \return        Exit status.
 */
int
transferOperator::filter(double tol)
{
  int status;
  
  /** Filter transition matrix (should change 2nd initDist with finalDist)*/
  if ((status = filterStochasticMatrix(P, initDist, initDist, tol, 2))) {
    fprintf(stderr, "transferOperator::filter: error filtering\
transition matrix.\n");
    throw std::exception();
  }

  return 0;
}


/**
 * Print transition matrix to file in coordinate format with header
 * (see transferOperator.hpp)
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printTransition(const char *path,
				  const char *fileFormat="txt",
				  const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printTransition, \
opening stream to write");
    }

  // Print
  if (strcmp(fileFormat, "bin") == 0)
    gsl_spmatrix_fwrite(fp, P);
  else
    gsl_spmatrix_fprintf(fp, P, dataFormat);
  
  if (ferror(fp))
    {
      throw std::ios::failure("transferOperator::printTransition, \
printing to transition matrix");
    }
  
  // Close
  fclose(fp);

  return 0;
}

/**
 * Print initial distribution to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printInitDist(const char *path,
				const char *fileFormat="txt",
				const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printInitDist, \
opening stream to write");
    }

  // Print
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_fwrite(fp, initDist);
  else
    gsl_vector_fprintf(fp, initDist, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print mask to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printMask(const char *path,
			    const char *fileFormat="txt",
			    const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printMask, \
opening stream to write");
    }
  
  // Print
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_uint_fwrite(fp, mask);
  else
    gsl_vector_uint_fprintf(fp, mask, dataFormat);
  
  // Close
  fclose(fp);

  return 0;
}

/**
 * Scan transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanTransition(const char *path,
				 const char *fileFormat="txt")
{
  FILE *fp;
  gsl_spmatrix *T;
  
  /** Open file */
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanTransition, \
opening stream to read");
  }

  /** Scan, summing if duplicate */
  if (strcmp(fileFormat, "bin") == 0)
    {
      /** Allocate before to read in CRS binary format */
      P = gsl_spmatrix_alloc2read(fp, GSL_SPMATRIX_CRS);
      gsl_spmatrix_fread(fp, P);
    }
  else
    {
      /** Read in Matrix Market format */
      T = gsl_spmatrix_fscanf(fp);
      if (!T)
	{
	  throw std::ios::failure("transferOperator::scanTransition, \
scanning transition matrix");
	}

      /** Compress */
      P = gsl_spmatrix_crs(T);

      /** Free */
      gsl_spmatrix_free(T);
    }

  /** Check if transition matrix has been properly read */
  if (!P)
    {
      fprintf(stderr, "transferOperator::scanTransition: error\
compressing transition matrix.\n");
      throw std::exception();
    }

  /** Set number of filled boxes in case not known already */
  NFilled = P->size1;

  /** Close */
  fclose(fp);
  
  return 0;
}

/**
 * Scan initial distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanInitDist(const char *path,
			       const char *fileFormat="txt")
{
  FILE *fp;
    
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanInitDist, \
opening stream to read");
    }

  /** Scan after preallocating */
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_fread(fp, initDist);
  else
    gsl_vector_fscanf(fp, initDist);
  
  //Close
  fclose(fp);
  
  return 0;
}


/**
 * Scan mask from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanMask(const char *path,
			   const char *fileFormat="txt")
{
  FILE *fp;
    
  // Open file
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanMask, \
opening stream to read");
    }

  /** Scan after preallocating */
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_uint_fread(fp, mask);
  else
    gsl_vector_uint_fscanf(fp, mask);
  
  //Close
  fclose(fp);
  
  return 0;
}


/** 
 * Add transition to count matrix from a pair of initial and final grid boxes.
 * \param[in]  T         Transition count matrix.
 * \param[in]  box0      Initial box of transition.
 * \param[in]  boxf      Final box of transition.
 * \return               Number of trajectories inside domain.
 */
size_t
transferOperator::addTransition(gsl_spmatrix *T, size_t box0, size_t boxf)
{
  size_t box0r, boxfr;
  size_t nIn = 0;
  double *ptr;

  /** Record transitions */
  /** Add transition triplet, summing if duplicate */
  if ((box0 < N) && (boxf < N))
    {
      /** Convert to reduced indices */
      box0r = gsl_vector_uint_get(mask, box0);
      boxfr = gsl_vector_uint_get(mask, boxf);

      /** Add triplet in reduced indices */
      ptr = gsl_spmatrix_ptr(T, box0r, boxfr);
      if (ptr)
	*ptr += 1.; /* sum duplicate values */
      else
	gsl_spmatrix_set(T, box0r, boxfr, 1.);   /* initalize to x */

      /** Add initial box to initial distribution (not on a reduced grid). */
      gsl_vector_set(initDist, box0r, gsl_vector_get(initDist, box0r) + 1.);

      nIn = 1;
    }

  return nIn;
}


/** 
 * Get the triplet vector counting the transitions
 * from pairs of grid boxes from the grid membership matrix.
 * \param[in]  gridMem   Grid membership matrix.
 * \param[out] T         Triplet matrix counting the transitions.
 * \return               Number of trajectories inside domain.
 */
size_t
transferOperator::getTransitionCountTriplet(const gsl_matrix_uint *gridMem,
					    gsl_spmatrix *T)
{
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  size_t nIn = 0;

  /** Initialize distributions to zero */
  if (initDist)
    gsl_vector_set_zero(initDist);

  /** Record transitions */
  for (size_t traj = 0; traj < nTraj; traj++)
    {
      /** Get initial and final boxes */
      box0 = gsl_matrix_uint_get(gridMem, traj, 0);
      boxf = gsl_matrix_uint_get(gridMem, traj, 1);
      
      /** Add transition triplet, summing if duplicate */
      nIn += addTransition(T, box0, boxf);
    }

  return nIn;
}


/*
 * Function definitions
 */


/** 
 * Get mask from grid membership matrix.
 * The mask is an N-dimensional vector giving the indices
 * of each box in the reduced set non-empty boxes.
 * If the grid membership matrix is NULL, no box is masked.
 * \param[out] mask    Mask.
 * \param[in]  gridMem Grid membership matrix.
 * \return     NFilled Number of filled boxes.
 */
size_t
getMask(gsl_vector_uint *mask, const gsl_matrix_uint *gridMem)
{
  const size_t N = mask->size;
  size_t nTraj;
  size_t NFilled;
  size_t box0, boxf;

  if (gridMem)
    {
      nTraj = gridMem->size1;

      // Initialized all boxes as empty (as marked by N)
      gsl_vector_uint_set_all(mask, N);

      // Loop over the states to check which box is filled
      for (size_t traj = 0; traj < nTraj; traj++)
	{
	  /** Get initial and final boxes */
	  box0 = gsl_matrix_uint_get(gridMem, traj, 0);
	  boxf = gsl_matrix_uint_get(gridMem, traj, 1);

	  // Flag initial and final boxes only if both are found
	  // (otherwise the transition will not be count by getTransitionCountTriplet)n
	  if ((box0 < N) && (boxf < N))
	    {
	      gsl_vector_uint_set(mask, box0, 1);
	      gsl_vector_uint_set(mask, boxf, 1);
	    }
	}

      // Loop over the boxes to finish the mask
      NFilled = 0;
      for (size_t box = 0; box < N; box++)
	{
	  if (gsl_vector_uint_get(mask, box) == 1)
	    {
	      gsl_vector_uint_set(mask, box, NFilled);
	      NFilled++;
	    }
	}
    }
  else
    {
      for (size_t box = 0; box < N; box++)
	gsl_vector_uint_set(mask, box, box);
      NFilled = N;
    }
  
  return NFilled;
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
