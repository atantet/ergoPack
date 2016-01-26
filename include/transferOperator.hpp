#ifndef TRANSFEROPERATOR_HPP
#define TRANSFEROPERATOR_HPP

#include <iostream>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spmatrix.h>
#if defined (WITH_OMP) && WITH_OMP == 1
#include <omp.h>
#endif

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

/** \brief Grid class.
 *
 * Grid class used for the Galerin approximation of transfer operators
 * by a transition matrix on a grid.
 */
class Grid {
  /** \brief Allocate memory. */
  void allocate(gsl_vector_uint *nx_);
  /** \brief Get uniform rectangular. */
  void getRectGrid(gsl_vector_uint *nx_,
		   const gsl_vector *xmin,
		   const gsl_vector *xmax);

public:
  /** Number of dimensions */
  size_t dim;
  /** Number of grid boxes */
  size_t N;
  /** Number of grid boxes per dimension */
  gsl_vector_uint *nx;
  /** Grid box bounds for each dimension */
  std::vector<gsl_vector *> *gridBounds;

  /** \brief Default constructor. */
  Grid(){}
  /** \brief Constructor allocating an empty grid. */
  Grid(gsl_vector_uint *nx_){ allocate(nx_); }
  /** \brief Construct a uniform rectangular grid with different dimensions. */
  Grid(gsl_vector_uint *nx_, const gsl_vector *xmin, const gsl_vector *xmax);
  /** \brief Construct a uniform rectangular grid with same dimensions. */
  Grid(size_t dim_, size_t inx, double xmin, double xmax);
  /** \brief Destructor. */
  ~Grid();
  
  /** \brief Print the grid to file. */
  int printGrid(const char *path, const char *dataFormat, bool verbose);
};


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

  /** \brief Allocate memory. */
  int allocate(size_t gridSize);
  /** \brief Get the transition matrices from a grid membership matrix. */
  int buildFromMembership(const gsl_matrix_uint *gridMem);

  
public:
  size_t N;              //!< Size of the grid
  gsl_spmatrix *P;       //!< Forward transition matrix (CRS)
  gsl_spmatrix *Q;       //!< Backward transition matrix (CRS)
  gsl_vector *rho0;      //!< Initial distribution
  gsl_vector *rhof;      //!< Final distribution

  
  /** \brief Default constructor */
  transferOperator(){}
  /** \brief Empty constructor allocating for grid size*/
  transferOperator(size_t gridSize){ allocate(gridSize); }
  /** \brief Constructor from the membership matrix. */
  transferOperator(const gsl_matrix_uint *gridMem, size_t gridSize);
  /** \brief Constructor from initial and final states for a given grid */
  transferOperator(const gsl_matrix *initStates, const gsl_matrix *finalStates,
		   const Grid *grid);
  /** \brief Constructor from a long trajectory for a given grid and lag */
  transferOperator(const gsl_matrix *states, const Grid *grid, size_t tauStep);
  /** \brief Destructor */
  ~transferOperator();

  
  /** \brief Filtering of weak Markov states. */
  int filter(double tol);


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
};



/*
 *  Functions declarations
 */

/** \brief Get membership matrix from initial and final states for a grid. */
gsl_matrix_uint *getGridMemMatrix(const gsl_matrix *initStates,
				  const gsl_matrix *finalStates,
				  const Grid *grid);
/** \brief Get the grid membership vector from a single long trajectory. */
gsl_matrix_uint *getGridMemMatrix(const gsl_matrix *states, const Grid *grid,
				  const size_t tauStep);
/** \brief Get the grid membership matrix from a single long trajectory. */
gsl_vector_uint *getGridMemVector(const gsl_matrix *states, const Grid *grid);
/** \brief Get the grid membership matrix from the membership vector for a given lag. */
gsl_matrix_uint *memVector2memMatrix(const gsl_vector_uint *gridMemVect, size_t tauStep);
/** \brief Concatenate a list of membership vectors into one membership matrix. */
gsl_matrix_uint * memVectorList2memMatrix(const std::vector<gsl_vector_uint *> *memList,
					  size_t tauStep);
/** \brief Get membership to a grid box of a single realization. */
int getBoxMembership(const gsl_vector *state, const Grid *grid);
/** \brief Get triplet vector from membership matrix. */
gsl_spmatrix *getTransitionCountTriplet(const gsl_matrix_uint *gridMem, size_t N);
/** \brief Remove weak nodes from a transition matrix. */
int filterStochasticMatrix(gsl_spmatrix *M, gsl_vector *rowCut, gsl_vector *colCut,
			   double tol, int norm);
/** \brief Normalize vector by the sum of its elements. */
void gsl_vector_normalize(gsl_vector *v);
/** \brief Get sum of vector elements. */
double gsl_vector_get_sum(gsl_vector *v);


/*
 * Constructors and destructors definitions
 */

/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem        GSL grid membership matrix.
 * \param[in] gridSize       Number of grid boxes.
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem, size_t gridSize)
{
  // Set grid size
  N = gridSize;

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
{
  gsl_matrix_uint *gridMem;

  // Set grid size
  N = grid->N;

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
{
  gsl_matrix_uint *gridMem;

  // Set grid size
  N = grid->N;

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

/**
 * Construct a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx_        GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 */
Grid::Grid(gsl_vector_uint *nx_, const gsl_vector *xmin, const gsl_vector *xmax)
{
  // Allocate and build uniform rectangular grid
  getRectGrid(nx_, xmin, xmax);
}

/**
 * Construct a uniform rectangular grid with same bounds for each dimension.
 * \param[in] dim_        Number of dimensions.
 * \param[in] inx         Number of boxes, identically for each dimension.
 * \param[in] dxmin       Minimum box limit, identically for each dimension.
 * \param[in] dxmax       Maximum box limit, identically for each dimension.
 */
Grid::Grid(size_t dim_, size_t inx, double dxmin, double dxmax)
{
  // Convert to uniform vectors to call getRectGrid.
  gsl_vector_uint *nx_ = gsl_vector_uint_alloc(dim_);
  gsl_vector *xmin_ = gsl_vector_alloc(dim_);
  gsl_vector *xmax_ = gsl_vector_alloc(dim_);
  gsl_vector_uint_set_all(nx_, inx);
  gsl_vector_set_all(xmin_, dxmin);
  gsl_vector_set_all(xmax_, dxmax);

  // Allocate and build uniform rectangular grid
  getRectGrid(nx_, xmin_, xmax_);

  // Free
  gsl_vector_uint_free(nx_);
  gsl_vector_free(xmin_);
  gsl_vector_free(xmax_);
}

/** Destructor desallocates memory used by the grid. */
Grid::~Grid()
{
  gsl_vector_uint_free(nx);
  for (size_t d = 0; d < dim; d++)
    gsl_vector_free((*gridBounds)[d]);
  delete gridBounds;
}


/*
 * Methods definitions
 */

/**
 * Allocate memory for the transition matrices and distributions.
 * \param[in] gridSize Number of grid boxes.
 * \return             Exit status.
 */
int
transferOperator::allocate(size_t gridSize)
{
  N = gridSize;
  
  P = gsl_spmatrix_alloc(N, N);
  if (!P) {
    fprintf(stderr, "Error allocating forward transition matrix.\n");
    return EXIT_FAILURE;
  }
  
  Q = gsl_spmatrix_alloc(N, N);
  if (!Q) {
    fprintf(stderr, "Error allocating backward transition matrix.\n");
    return EXIT_FAILURE;
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
  if (!(T = getTransitionCountTriplet(gridMem, N))) {
    fprintf(stderr, "Error getting transition triplet matrix.");
    return EXIT_FAILURE;
  }
  
  /** Convert to CRS summing duplicates */
  if (!(P = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS))) {
    fprintf(stderr, "Error converting transition triplet to CRS.\n");
    return EXIT_FAILURE;
  }

  /** Get transpose copy */
  if (!(Q = gsl_spmatrix_alloc_nzmax(N, N, P->nz, GSL_SPMATRIX_CRS))) {
    fprintf(stderr, "Error allocating backward transition matrix \
before transpose copy.\n");
    return EXIT_FAILURE;
  }      
  if (gsl_spmatrix_transpose_memcpy(Q, P)) {
    fprintf(stderr, "Error transposing transition matrix.\n");
    return EXIT_FAILURE;
  }
  
  /** Get initial distribution */
  if (!(rho0 = gsl_spmatrix_get_colsum(Q))) {
    fprintf(stderr, "Error getting initial distribution.\n");
    return EXIT_FAILURE;
  }

  /** Get final distribution */
  if (!(rhof = gsl_spmatrix_get_colsum(P))) {
    fprintf(stderr, "Error getting final distribution.\n");
    return EXIT_FAILURE;
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
  if (status = filterStochasticMatrix(P, rho0, rhof, tol, 2)) {
    fprintf(stderr, "Error while filtering forward transition matrix.\n");
    return EXIT_FAILURE;
  }

  /** Filter forward transition matrix */
  if (status = filterStochasticMatrix(Q, rhof, rho0, tol, 2)) {
    fprintf(stderr, "Error while filtering backward transition matrix.\n");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "w"))){
    fprintf(stderr, "Can't open %s for writing forward transition matrix: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  // Print
  gsl_spmatrix_fprintf(fp, P, dataFormat);
  if (ferror(fp)) {
    fprintf(stderr, "Error while printing forward transition matrix.\n");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "w"))) {
    fprintf(stderr, "Can't open %s for writing backward transition matrix: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  // Print 
  gsl_spmatrix_fprintf(fp, Q, dataFormat);
  if (ferror(fp)) {
    fprintf(stderr, "Error while printing backward transition matrix.\n");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "w"))){
    fprintf(stderr, "Can't open %s for writing initial distribution: ", path);
    perror("");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "w"))) {
    fprintf(stderr, "Can't open %s for writing final distribution: ", path);
    perror("");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "r"))){
    fprintf(stderr, "Can't open %s for reading forward transition matrix: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  /** Scan, summing if duplicate */
  if (!(T = gsl_spmatrix_fscanf(fp, 1))) {
    fprintf(stderr, "Error while scanning forward transition matrix.\n");
    return EXIT_FAILURE;
  }

  /** Compress */
  P = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS);
  gsl_spmatrix_free(T);
  if (!P) {
    fprintf(stderr, "Error converting transition triplet to CRS.\n");
    return EXIT_FAILURE;
  }

  /** Check if matrix dimensions consistent with grid size */
  if ((P->size1 != P->size2) || (P->size1 != N)) {
    fprintf(stderr, "Error: forward transition matrix dimensions inconsistent with grid\n");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "r"))) {
    fprintf(stderr, "Can't open %s for reading backward transition matrix: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  /** Scan, summing if duplicate */
  if (!(T = gsl_spmatrix_fscanf(fp, 1))) {
    fprintf(stderr, "Error while scanning backward transition matrix.\n");
    return EXIT_FAILURE;
  }

  /** Compress */
  Q = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS);
  gsl_spmatrix_free(T);
  if (!Q) {
    fprintf(stderr, "Error converting transition triplet to CRS.\n");
    return EXIT_FAILURE;
  }

  /** Check if matrix dimensions consistent with grid size */
  if ((Q->size1 != Q->size2) || (Q->size1 != N)) {
    fprintf(stderr, "Error: backward transition matrix dimensions inconsistent with grid\n");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "r"))) {
    fprintf(stderr, "Can't open %s for reading initial distribution: ", path);
    perror("");
    return EXIT_FAILURE;
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
  if (!(fp = fopen(path, "r"))){
    fprintf(stderr, "Can't open %s for reading final distribution: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  /** Scan after preallocating */
  rhof = gsl_vector_alloc(N);
  gsl_vector_fscanf(fp, rhof);
  
  //Close
  fclose(fp);
  
  return 0;
}


/**
 * Allocate memory for the grid.
 * \param[in] GSL vector of unsigned integers giving the number of boxes per dimension.
 */
void
Grid::allocate(gsl_vector_uint *nx_)
{
  dim = nx_->size;
  
  nx = gsl_vector_uint_alloc(dim);
  gsl_vector_uint_memcpy(nx, nx_);
  
  N = 1;
  gridBounds = new std::vector<gsl_vector *>(dim);
  for (size_t d = 0; d < dim; d++){
    N *= gsl_vector_uint_get(nx, d);
    (*gridBounds)[d] = gsl_vector_alloc(gsl_vector_uint_get(nx, d) + 1);
  }

  return;
}

/**
 * Get a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx         GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 */
void
Grid::getRectGrid(gsl_vector_uint *nx_, const gsl_vector *xmin, const gsl_vector *xmax)
{
  double delta;
  
  // Allocate
  allocate(nx_);

  // Build uniform grid bounds
  for (size_t d = 0; d < dim; d++) {
    // Get spatial step
    delta = (gsl_vector_get(xmax, d) - gsl_vector_get(xmin, d))
      / gsl_vector_uint_get(nx, d);
    // Set grid bounds
    gsl_vector_set((*gridBounds)[d], 0, gsl_vector_get(xmin, d));
    for (size_t i = 1; i < gsl_vector_uint_get(nx, d) + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return;
}

/**
 * Print the grid to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] dataFormat Format in which to print each element.
 * \param[in] verbose    If true, also print to the standard output.  
 * \return               Status.
 */
int
Grid::printGrid(const char *path, const char *dataFormat="%lf", bool verbose=false)
{
  gsl_vector *bounds;
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w"))){
    fprintf(stderr, "Can't open %s for printing grid: ", path);
    perror("");
    return EXIT_FAILURE;
  }

  if (verbose)
    std::cout << "Domain grid (min, max, n):" << std::endl;

  // Print grid
  for (size_t d = 0; d < dim; d++) {
    bounds = (*gridBounds)[d];
    if (verbose) {
      std::cout << "dim " << d+1 << ": ("
		<< gsl_vector_get(bounds, 0) << ", "
		<< gsl_vector_get(bounds, bounds->size - 1) << ", "
		<< (bounds->size - 1) << ")" << std::endl;
    }
    
    for (size_t i = 0; i < bounds->size; i++){
      fprintf(fp, dataFormat, gsl_vector_get((*gridBounds)[d], i));
      fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }

  /** Check for printing errors */
  if (ferror(fp)) {
    fprintf(stderr, "Error printing grid to file %s\n", path);
    return EXIT_FAILURE;
  }

  // Close
  fclose(fp);

  return 0;
}


/*
 * Function definitions
 */

/**
 * Get membership matrix from initial and final states for a grid.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] grid           Pointer to Grid object.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMemMatrix(const gsl_matrix *initStates, const gsl_matrix *finalStates,
		 const Grid *grid)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  gsl_matrix_uint *gridMem;

  // Allocate
  gridMem = gsl_matrix_uint_alloc(grid->N, 2);
  
  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nTraj; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, initStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 0, getBoxMembership(X, grid));
      
      // Find final box
      gsl_matrix_get_row(X, finalStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 1, getBoxMembership(X, grid));
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}

/**
 * Get the grid membership vector from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] grid           Pointer to Grid object.
 * \return                   GSL grid membership vector.
 */
gsl_vector_uint *
getGridMemVector(const gsl_matrix *states, const Grid *grid)
{
  const size_t nStates = states->size1;
  const size_t dim = states->size2;
  gsl_vector_uint *gridMem = gsl_vector_uint_alloc(nStates);

  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nStates; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, states, traj);
#pragma omp critical
      {
	gsl_vector_uint_set(gridMem, traj, getBoxMembership(X, grid));
      }
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}

/**
 * Get the grid membership matrix from the membership vector for a given lag.
 * \param[in] gridMemVect    Grid membership vector of a long trajectory for a grid.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
memVector2memMatrix(gsl_vector_uint *gridMemVect, size_t tauStep)
{
  const size_t nStates = gridMemVect->size;
  gsl_matrix_uint *gridMem = gsl_matrix_uint_alloc(nStates - tauStep, 2);

  // Get membership matrix from vector
  for (size_t traj = 0; traj < (nStates - tauStep); traj++) {
    gsl_matrix_uint_set(gridMem, traj, 0,
		       gsl_vector_uint_get(gridMemVect, traj));
    gsl_matrix_uint_set(gridMem, traj, 1,
		       gsl_vector_uint_get(gridMemVect, traj + tauStep));
  }

  return gridMem;
}

/**
 * Get the grid membership matrix from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] grid           Pointer to Grid object.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMemMatrix(const gsl_matrix *states, const Grid *grid, const size_t tauStep)
{
  // Get membership vector
  gsl_vector_uint *gridMemVect = getGridMemVector(states, grid);

  // Get membership matrix from vector
  gsl_matrix_uint *gridMem = memVector2memMatrix(gridMemVect, tauStep);

  // Free
  gsl_vector_uint_free(gridMemVect);
  
  return gridMem;
}

/**
 * Concatenate a list of membership vectors into one membership matrix.
 * \param[in] memList    STD vector of membership GSL vectors each of them associated
 * with a single long trajectory.
 * \param[in] tauStep    Lag used to calculate the transitions.
 * \return               GSL grid membership matrix.
 */
gsl_matrix_uint *
memVectorList2memMatrix(const std::vector<gsl_vector_uint *> *memList, size_t tauStep)
{
  size_t nStatesTot = 0;
  size_t count;
  const size_t listSize = memList->size();
  gsl_matrix_uint *gridMem, *gridMemMatrixL;

  // Get total number of states and allocate grid membership matrix
  for (size_t l = 0; l < listSize; l++)
    nStatesTot += (memList->at(l))->size;
  gridMem = gsl_matrix_uint_alloc(nStatesTot - tauStep * listSize, 2);
  
  // Get membership matrix from list of membership vectors
  count = 0;
  for (size_t l = 0; l < listSize; l++) {
    gridMemMatrixL = memVector2memMatrix(memList->at(l), tauStep);
    for (size_t t = 0; t < gridMemMatrixL->size1; t++) {
      gsl_matrix_uint_set(gridMem, count, 0,
			  gsl_matrix_uint_get(gridMemMatrixL, t, 0));
      gsl_matrix_uint_set(gridMem, count, 1,
			  gsl_matrix_uint_get(gridMemMatrixL, t, 1));
      count++;
    }
    gsl_matrix_uint_free(gridMemMatrixL);
  }
  
  return gridMem;
}

/**
 * Get membership to a grid box of a single realization.
 * \param[in] state          GSL vector of a single state.
 * \param[in] grid           Pointer to Grid object.
 * \return                   Box index to which the state belongs.
 */
int
getBoxMembership(const gsl_vector *state, const Grid *grid)
{
  const size_t dim = state->size;
  size_t inBox, nBoxDir;
  size_t foundBox;
  size_t subbp, subbn, ids;
  gsl_vector *bounds;

  // Get box
  foundBox = grid->N;
  for (size_t box = 0; box < grid->N; box++){
    inBox = 0;
    subbp = box;
    for (size_t d = 0; d < dim; d++){
      bounds = grid->gridBounds->at(d);
      nBoxDir = bounds->size - 1;
      subbn = (size_t) (subbp / nBoxDir);
      ids = subbp - subbn * nBoxDir;
      inBox += (size_t) ((gsl_vector_get(state, d)
			  >= gsl_vector_get(bounds, ids))
			 & (gsl_vector_get(state, d)
			    < gsl_vector_get(bounds, ids+1)));
      subbp = subbn;
    }
    if (inBox == dim){
      foundBox = box;
      break;
    }
  }
  
  return foundBox;
}

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
  if (!(T = gsl_spmatrix_alloc_nzmax(N, N, nTraj, GSL_SPMATRIX_TRIPLET))) {
    fprintf(stderr, "Error allocating transition triplet matrix.\n");
    return NULL;
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
  for (n = 0; n < M->size1; n++) {
    if (gsl_vector_get(rowCut, n) < tol) {
      gsl_vector_uint_set(rowOut, n, 1);
      gsl_vector_set(rowCut, n, 0.);
    }
  }
  colOut = gsl_vector_uint_calloc(M->size2);
  for (n = 0; n < M->size2; n++) {
    if (gsl_vector_get(colCut, n) < tol) {
      gsl_vector_uint_set(colOut, n, 1);
      gsl_vector_set(colCut, n, 0.);
    }
  }

  if (GSL_SPMATRIX_ISTRIPLET(M)) {
    fprintf(stderr, "Stochastic matrix to filter should not be triplet.\n");
    return EXIT_FAILURE;
  }
  else if (GSL_SPMATRIX_ISCCS(M)) {
    for (outerIdx = 0; outerIdx < M->outerSize; outerIdx++) {
      isColOut = gsl_vector_uint_get(colOut, outerIdx);
      for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; p++) {
	isRowOut = gsl_vector_uint_get(rowOut, M->i[p]);
	// Remove elements of states to be removed
	if (isRowOut || isColOut)
	  M->data[p] = 0.;
      }
    }
  }
  else if (GSL_SPMATRIX_ISCRS(M)) {
    for (outerIdx = 0; outerIdx < M->outerSize; outerIdx++) {
      isRowOut = gsl_vector_uint_get(rowOut, outerIdx);
      for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; p++) {
	isColOut = gsl_vector_uint_get(colOut, M->i[p]);
	// Remove elements of states to be removed
	if (isRowOut || isColOut)
	  M->data[p] = 0.;
      }
    }
  }
    
  /** Make matrix and vectors stochastic again */
  switch (norm){
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


/**
 * Normalize vector by the sum of its elements.
 * \param[input] v Vector to normalize.
 */
void
gsl_vector_normalize(gsl_vector *v)
{
  double sum = gsl_vector_get_sum(v);

  for (size_t j = 0; j < v->size; j++)
    v->data[j * v->stride] /= sum;
  
  return;
}

/**
 * Get sum of vector elements.
 * \param[input] v Vector from which to sum the elements.
 * \return         Sum of vector elements.
 */
double
gsl_vector_get_sum(gsl_vector *v)
{
  double sum = 0;

  for (size_t j = 0; j < v->size; j++)
    sum += v->data[j * v->stride];
  
  return sum;
}


#endif
