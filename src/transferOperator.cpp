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
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem     GSL grid membership matrix.
 * \param[in] N_          Number of grid boxes.
 * \param[in] stationary_ Whether the system is stationary
 *                        (in which case \f$\rho_0 = \rho_f\f$ and no need
 *                        to calculate the backward transition matrix).
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem,
				   const size_t N_,
				   const bool stationary_=false)
  : N(N_), stationary(stationary_)
{
  // Get mask
  mask = gsl_vector_uint_alloc(N);
  NFilled = getMask(gridMem, mask);

  // Allocate distributions and set matrices to NULL pointer
  allocate();

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
				   const Grid *grid,
				   const bool stationary_=false)
  : N(grid->getN()), stationary(stationary_)
{
  gsl_matrix_uint *gridMem;

  // Get grid membership matrix
  gridMem = getGridMemMatrix(initStates, finalStates, grid);

  // Get mask
  mask = gsl_vector_uint_alloc(N);
  NFilled = getMask(gridMem, mask);

  // Allocate distributions and set matrices to NULL pointer
  allocate();

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);  
}

/**
 * Construct transferOperator calculating the forward
 * and backward transition matrices
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

  // Get mask
  mask = gsl_vector_uint_alloc(N);  
  NFilled = getMask(gridMem, mask);

  // Allocate distributions and set matrices to NULL pointer
  allocate();

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
  initDist = gsl_vector_alloc(NFilled);
  
  /** If stationary problem, initDist = finalDist = stationary distribution
   *  and the backward transition matrix need not be calculated */
  Q = NULL;
  if (!stationary)
      finalDist = gsl_vector_alloc(NFilled);
  else
      finalDist = initDist;
  
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
  const double tol = .9 / nTraj; // The min non-zero dist is 1. / nTraj
  size_t nIn;
  gsl_spmatrix *T;

  // Get transition count triplets
  if (!(T = gsl_spmatrix_alloc_nzmax(NFilled, NFilled, nTraj,
				     GSL_SPMATRIX_TRIPLET)))
    {
      fprintf(stderr, "buildFromMembership: error allocating\
triplet count matrix.\n");
      std::bad_alloc();
    }

  if (stationary)
    nIn = getTransitionCountTriplet(gridMem, mask, T, initDist, NULL);
  else
    nIn = getTransitionCountTriplet(gridMem, mask, T, initDist, finalDist);
  std::cout <<  nIn * 100. / nTraj
	    << "% of the trajectories ended up inside the domain." << std::endl;

  
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
      if (!(Q = gsl_spmatrix_alloc_nzmax(NFilled, NFilled, P->nz, GSL_SPMATRIX_CRS)))
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
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printForwardTransition(const char *path,
					 const char *fileFormat="txt",
					 const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printForwardTransition, \
opening stream to write");
    }

  // Print
  if (strcmp(fileFormat, "bin") == 0)
    gsl_spmatrix_fwrite(fp, P);
  else
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
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printBackwardTransition(const char *path,
					  const char *fileFormat="txt",
					  const char *dataFormat="%lf")
{
  FILE *fp;

  // Print
  if (Q)
    {
      // Open file
      if (!(fp = fopen(path, "w")))
	{
	  throw std::ios::failure("transferOperator::printBackwardTransition, \
opening stream to write");
	}

      // Print 
      if (strcmp(fileFormat, "bin") == 0)
	gsl_spmatrix_fwrite(fp, Q);
      else
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
 * Print final distribution to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \param[in] dataFormat Format in which to print each element (if formatted output).
 * \return               Status.
 */
int
transferOperator::printFinalDist(const char *path,
				 const char *fileFormat="txt",
				 const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      throw std::ios::failure("transferOperator::printFinalDist, \
opening stream to write");
    }
  
  // Print
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_fwrite(fp, finalDist);
  else
    gsl_vector_fprintf(fp, finalDist, dataFormat);
  
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
 * Scan forward transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanForwardTransition(const char *path,
					const char *fileFormat="txt")
{
  FILE *fp;
  gsl_spmatrix *T;
  
  /** Open file */
  if (!(fp = fopen(path, "r")))
    {
      throw std::ios::failure("transferOperator::scanForwardTransition, \
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
	  throw std::ios::failure("transferOperator::scanForwardTransition, \
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
      fprintf(stderr, "transferOperator::scanForwardTransition: error\
compressing forward transition matrix.\n");
      throw std::exception();
    }

  /** Close */
  fclose(fp);
  
  return 0;
}

/**
 * Scan backward transition matrix from file in coordinate format with header.
 * (see transferOperator.hpp).
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanBackwardTransition(const char *path,
					 const char *fileFormat="txt")
{
  FILE *fp;
  gsl_spmatrix *T;
  
  
  /** Open file */
  if (!stationary)
    {
      if (!(fp = fopen(path, "r")))
	{
	  throw std::ios::failure("transferOperator::scanBackwardTransition, \
opening stream to read");
	}

      /** Scan, summing if duplicate */
      if (strcmp(fileFormat, "bin") == 0)
	{
	  /** Allocate before to read in CRS binary format */
	  Q = gsl_spmatrix_alloc2read(fp, GSL_SPMATRIX_CRS);
	  gsl_spmatrix_fread(fp, Q);
	}
      else
	{
	  /** Read in Matrix Market format */
	  T = gsl_spmatrix_fscanf(fp);
	  if (!T)
	    {
	      throw std::ios::failure("transferOperator::scanForwardTransition, \
scanning transition matrix");
	    }

	  /** Compress */
	  Q = gsl_spmatrix_crs(T);

	  /** Free */
	  gsl_spmatrix_free(T);
	}

      /** Check if transition matrix has been properly read */
      if (!Q)
	{
	  fprintf(stderr, "transferOperator::scanBackwardTransition: error\
compressing backward transition matrix.\n");
	  throw std::exception();
	}
      
      /** Close */
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
 * Scan final distribution from file.
 * No preliminary memory allocation needed but the grid size should be set.
 * Previously allocated memory should first be freed to avoid memory leak.
 * \param[in] path       Path to the file in which to scan.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 * \return               0 in success, EXIT_FAILURE otherwise.
 */
int
transferOperator::scanFinalDist(const char *path,
				const char *fileFormat="txt")
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
      if (strcmp(fileFormat, "bin") == 0)
	gsl_vector_fread(fp, finalDist);
      else
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


/*
 * Function definitions
 */

/** 
 * Get the triplet vector counting the transitions
 * from pairs of grid boxes from the grid membership matrix.
 * \param[in]  gridMem   Grid membership matrix.
 * \param[in]  mask      Mask.
 * \param[out] T         Triplet matrix counting the transitions.
 * \param[out] initDist  Initial states count
 * \param[out] finalDist Final states count
 * \return               Number of trajectories inside domain.
 */
size_t
getTransitionCountTriplet(const gsl_matrix_uint *gridMem,
			  const gsl_vector_uint *mask,
			  gsl_spmatrix *T, gsl_vector *initDist=NULL,
			  gsl_vector *finalDist=NULL)
{
  const size_t nTraj = gridMem->size1;
  const size_t N = mask->size;
  size_t box0, boxf, box0r, boxfr;
  size_t nIn = 0;
  double *ptr;

  /** Initialize distributions to zero */
  if (initDist)
    gsl_vector_set_zero(initDist);
  if (finalDist)
    gsl_vector_set_zero(finalDist);

  /** Record transitions */
  for (size_t traj = 0; traj < nTraj; traj++)
    {
      /** Get initial and final boxes */
      box0 = gsl_matrix_uint_get(gridMem, traj, 0);
      boxf = gsl_matrix_uint_get(gridMem, traj, 1);
      
      /** Convert to reduced indices */
      box0r = gsl_vector_uint_get(mask, box0);
      boxfr = gsl_vector_uint_get(mask, boxf);
    
      /** Add transition triplet, summing if duplicate */
      if ((box0 < N) && (boxf < N))
	{
	  ptr = gsl_spmatrix_ptr(T, box0r, boxfr);
	  if (ptr)
	    *ptr += 1.; /* sum duplicate values */
	  else
	    gsl_spmatrix_set(T, box0r, boxfr, 1.);   /* initalize to x */

	  /** Add initial and final boxes to initial and final distributions,
	      respectively (not on a reduced grid). */
	  if (initDist)
	    gsl_vector_set(initDist, box0r, gsl_vector_get(initDist, box0r) + 1.);
	  if (finalDist)
	    gsl_vector_set(finalDist, boxfr, gsl_vector_get(finalDist, boxfr) + 1.);

	  nIn++;
	}
    }

  return nIn;
}


/** 
 * Get mask from grid membership matrix.
 * The mask is an N-dimensional vector giving the indices
 * of each box in the reduced set non-empty boxes.
 * \param[in]  gridMem Grid membership matrix.
 * \param[out] mask    Mask.
 * \return     NFilled Number of filled boxes.
 */
size_t
getMask(const gsl_matrix_uint *gridMem, gsl_vector_uint *mask)
{
  const size_t N = mask->size;
  const size_t nTraj = gridMem->size1;
  size_t NFilled;

  // Initialized all boxes as empty (as marked by N)
  gsl_vector_uint_set_all(mask, N);

  // Loop over the states to check which box is filled
  for (size_t traj = 0; traj < nTraj; traj++)
    {
      // Flag initial box
      gsl_vector_uint_set(mask, gsl_matrix_uint_get(gridMem, traj, 0), 1);
      // Flag final box
      gsl_vector_uint_set(mask, gsl_matrix_uint_get(gridMem, traj, 1), 1);
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
