#include <ergoGrid.hpp>

/** \file ergoGrid.cpp
 *  \brief Definitions for ergoGrid.hpp
 *
 */

/*
 * Constructors and destructors definitions
 */

/**
 * Construct a uniform rectangular grid with same dimensions
 * adapted to the time series.
 * \param[in] inx        Number of boxes for each dimension.
 * \param[in] nSTDLow    Number of standard deviations
 *                       to span away from the mean state from below.
 * \param[in] nSTDHigh   Number of standard deviations
 *                       to span away from the mean state from above.
 * \param[in] states     Time series.
 */
RegularGrid::RegularGrid(const size_t inx,
			 const double nSTDLow, const double nSTDHigh,
			 const gsl_matrix *states)
  : CurvilinearGrid(states->size2, inx)
{
  gsl_vector *nSTDLowv, *nSTDHighv;
  
  // Allocate memory for the grid bounds
  allocateBounds();
  
  // Get uniform vectors
  if (nSTDLow && nSTDHigh)
    {
      nSTDLowv = gsl_vector_alloc(dim);
      nSTDHighv = gsl_vector_alloc(dim);
      gsl_vector_set_all(nSTDLowv, nSTDLow);
      gsl_vector_set_all(nSTDHighv, nSTDHigh);
    }
  else
    {
      nSTDLowv = NULL;
      nSTDHighv = NULL;
    }

  // Get adapted grid
  getAdaptedRegularGrid(nSTDLowv, nSTDHighv, states);

  // Free
  if (nSTDLowv && nSTDHighv)
    {
      gsl_vector_free(nSTDLowv);
      gsl_vector_free(nSTDHighv);
    }
}


/**
 * Construct a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx_        Vector giving the number of boxes for each dimension.
 * \param[in] xmin       Vector giving the minimum box limit for each dimension.
 * \param[in] xmax       Vector giving the maximum box limit for each dimension.
 */
RegularGrid::RegularGrid(const gsl_vector_uint *nx_,
		   const gsl_vector *xmin, const gsl_vector *xmax)
  : CurvilinearGrid(nx_)
{
  // Allocate memory for the grid bounds
  allocateBounds();

  // Allocate and build uniform rectangular grid
  getRegularGrid(xmin, xmax);
}

/**
 * Construct a uniform rectangular grid with same bounds for each dimension.
 * \param[in] dim_        Number of dimensions.
 * \param[in] inx         Number of boxes, identically for each dimension.
 * \param[in] dxmin       Minimum box limit, identically for each dimension.
 * \param[in] dxmax       Maximum box limit, identically for each dimension.
 */
RegularGrid::RegularGrid(const size_t dim_, const size_t inx,
		   const double dxmin, const double dxmax)
  : CurvilinearGrid(dim_, inx)
{
  // Allocate memory for the grid bounds
  allocateBounds();

  // Convert to uniform vectors to call getRegularGrid.
  gsl_vector *xmin_ = gsl_vector_alloc(dim_);
  gsl_vector *xmax_ = gsl_vector_alloc(dim_);
  gsl_vector_set_all(xmin_, dxmin);
  gsl_vector_set_all(xmax_, dxmax);

  // Allocate and build uniform rectangular grid
  getRegularGrid(xmin_, xmax_);

  // Free
  gsl_vector_free(xmin_);
  gsl_vector_free(xmax_);
}

/**
 * Construct a uniform rectangular grid adapted to the time series.
 * \param[in] nx_        Vector giving the number of boxes for each dimension.
 * \param[in] nSTDLow    Vector giving the number of standard deviations
 *                       to span away from the mean state from below.
 * \param[in] nSTDHigh   Vector giving the number of standard deviations
 *                       to span away from the mean state from above.
 * \param[in] states     Time series.
 */
RegularGrid::RegularGrid(const gsl_vector_uint *nx_,
		   const gsl_vector *nSTDLow, const gsl_vector *nSTDHigh,
		   const gsl_matrix *states)
  : CurvilinearGrid(nx_)
{
  // Allocate memory for the grid bounds
  allocateBounds();

  // Get adapted grid
  getAdaptedRegularGrid(nSTDLow, nSTDHigh, states);
}

/*
 * Methods definitions
 */

/**
 * Get a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] xmin       Vector giving the minimum box limit for each dimension.
 * \param[in] xmax       Vector giving the maximum box limit for each dimension.
 */
void
RegularGrid::getRegularGrid(const gsl_vector *xmin, const gsl_vector *xmax)
{
  double delta;
  
  // Build uniform grid bounds
  for (size_t d = 0; d < dim; d++)
    {
      // Get spatial step
      delta = (gsl_vector_get(xmax, d) - gsl_vector_get(xmin, d))
	/ gsl_vector_uint_get(nx, d);
      
      // Set grid bounds
      gsl_vector_set((*bounds)[d], 0, gsl_vector_get(xmin, d));
      for (size_t i = 1; i < gsl_vector_uint_get(nx, d) + 1; i++)
	{
	  gsl_vector_set((*bounds)[d], i,
			 gsl_vector_get((*bounds)[d], i-1) + delta);
	}
    }

  return;
}

/**
 * Construct a uniform rectangular grid adapted to the time series.
 * \param[in] nSTDLow    Vector giving the number of standard deviations
 *                       to span away from the mean state from below.
 * \param[in] nSTDHigh   Vector giving the number of standard deviations
 *                       to span away from the mean state from above.
 * \param[in]state       Time series.
 */
void
RegularGrid::getAdaptedRegularGrid(const gsl_vector *nSTDLow, const gsl_vector *nSTDHigh,
				   const gsl_matrix *states)
{
  gsl_vector *xmin, *xmax, *statesMean, *statesSTD;
  
  xmin = gsl_vector_alloc(dim);
  xmax = gsl_vector_alloc(dim);

  // If then number of STD is given, get mean and std of time series
  // Otherwise, get minimum and maximum
  if (nSTDLow && nSTDHigh)
    {
      // Find mean and std
      statesMean = gsl_vector_alloc(states->size2);
      statesSTD = gsl_vector_alloc(states->size2);
      gsl_matrix_get_mean(statesMean, states, 0);
      gsl_matrix_get_std(statesSTD, states, 0);
      
      // Define limits as multiples of std around the mean
      for (size_t d = 0; d < dim; d++)
	{
	  gsl_vector_set(xmin, d, gsl_vector_get(statesMean, 0)
			 - gsl_vector_get(nSTDLow, d)
			 * gsl_vector_get(statesSTD, d));
	  gsl_vector_set(xmax, d, gsl_vector_get(statesMean, d)
			 + gsl_vector_get(nSTDHigh, d)
			 * gsl_vector_get(statesSTD, d));
	}
      
      // Free
      gsl_vector_free(statesMean);
      gsl_vector_free(statesSTD);
    }
  else
    {
      // Get min and max of time series
      gsl_matrix_get_min(xmin, states, 0);
      gsl_matrix_get_max(xmax, states, 0);
      // The upper bound uses a strict inequality,
      // make sure the maximum is included
      gsl_vector_add_constant(xmax, 1.e-12);
    }
  
  // Allocate and build uniform rectangular grid
  getRegularGrid(xmin, xmax);

  // Free
  gsl_vector_free(xmin);
  gsl_vector_free(xmax);
}

/**
 * Print the grid to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] dataFormat Format in which to print each element.
 * \param[in] verbose    If true, also print to the standard output.  
 * \return               Status.
 */
int
RegularGrid::printGrid(const char *path, const char *dataFormat="%lf",
		       const bool verbose=false) const
{
  gsl_vector *dimBounds;
  FILE *fp;

  // Open file
  if (!(fp = fopen(path, "w")))
    {
      std::ios::failure("Grid::printGrid, opening stream for writing");
    }

  if (verbose)
    std::cout << "Domain grid (min, max, n):" << std::endl;

  // Print grid
  for (size_t d = 0; d < dim; d++)
    {
      dimBounds = bounds->at(d);
      if (verbose)
      	{
      	  std::cout << "dim " << d+1 << ": ("
      		    << gsl_vector_get(dimBounds, 0) << ", "
      		    << gsl_vector_get(dimBounds, dimBounds->size - 1) << ", "
      		    << (dimBounds->size - 1) << ")" << std::endl;
      	}

      for (size_t i = 0; i < dimBounds->size; i++)
      	{
      	  fprintf(fp, dataFormat, gsl_vector_get(dimBounds, i));
      	  fprintf(fp, " ");
      	}
      fprintf(fp, "\n");
    }
  
  /** Check for printing errors */
  if (ferror(fp))
    {
      std::ios::failure("Grid::printGrid, printing grid");
    }
  
  // Close
  fclose(fp);
  
  return 0;
}

/**
 * Get membership to a grid box of a single realization.
 * \param[in] state          Vector of a single state.
 * \return                   Box index to which the state belongs.
 */
size_t
RegularGrid::getBoxMembership(const gsl_vector *state) const
{
  const size_t dim = state->size;
  size_t nFoundInt, nBoxDir, foundBox, idInt;
  gsl_vector *boundsDir;
  gsl_vector_uint *idIntervals = gsl_vector_uint_alloc(dim);

  // Get box
  foundBox = N;
  nFoundInt = 0;
  // Check for each dimension
  for (size_t d = 0; d < dim; d++)
    {
      // Get bounds for each interval of direction d
      boundsDir = bounds->at(d);
      // Get the number of intervals
      nBoxDir = gsl_vector_uint_get(nx, d);
      
      // If state outisde domain, no need to continue to test,
      // the box is not found.
      // The intervals must not overlap and be ordered increasingly.
      idInt = 0;
      if (gsl_vector_get(state, d) < gsl_vector_get(boundsDir, idInt))
	break;

      // Get interval for direction d
      while ((idInt < nBoxDir) &&
	     (gsl_vector_get(state, d) > gsl_vector_get(boundsDir, idInt + 1)))
	// Not in interval idInt, iterate to next interval
	idInt++;

      // Check if interval was found.
      // If yes, save the index for direction d and go to next direction.
      // Otherwise the box is outside the domain, no need to continue, the box is not found.
      if (idInt < nBoxDir)
	{
	  nFoundInt++;
	  gsl_vector_uint_set(idIntervals, d, idInt);
	  continue;
	}
      else
	break;
    }
  
  // Is the state in this box?
  if (nFoundInt == dim)
    {
      // Convert interval indices to box index, e.g.:
      // foundBox = iz + nz * (iy + ny * ix)
      foundBox = gsl_vector_uint_get(idIntervals, 0);
      for (size_t d = 1; d < dim; d++)
	foundBox = foundBox * gsl_vector_uint_get(nx, d)
	  + gsl_vector_uint_get(idIntervals, d);
    }

  // Free
  gsl_vector_uint_free(idIntervals);
  
  return foundBox;
}


// /**
//  * Get membership to a grid box of a single realization.
//  * \param[in] state          Vector of a single state.
//  * \return                   Box index to which the state belongs.
//  */
// size_t
// RegularGrid::getBoxMembership(const gsl_vector *state) const
// {
//   const size_t dim = state->size;
//   size_t inBox, nBoxDir;
//   size_t foundBox;
//   size_t subbp, subbn, ids;
//   gsl_vector *boundsDir;

//   // Get box
//   foundBox = N;
//   for (size_t box = 0; box < N; box++)
//     {
//       inBox = 0;
//       subbp = box;
//       for (size_t d = 0; d < dim - 1; d++)
// 	{
// 	  nBoxDir = gsl_vector_uint_get(nx, d);
// 	  subbn = (size_t) (subbp / nBoxDir);
// 	  ids = subbp - subbn * nBoxDir;
// 	  boundsDir = bounds->at(dim - 1 - d);
// 	  inBox += (size_t) ((gsl_vector_get(state, dim - 1 - d)
// 			      >= gsl_vector_get(boundsDir, ids))
// 			     & (gsl_vector_get(state, dim - 1 - d)
// 				< gsl_vector_get(boundsDir, ids+1)));
// 	  subbp = subbn;
// 	}
//       // Last dimension, the index is directly given by subbp
//       ids = subbp;
//       boundsDir = bounds->at(0);
//       inBox += (size_t) ((gsl_vector_get(state, 0)
// 			  >= gsl_vector_get(boundsDir, ids))
// 			 & (gsl_vector_get(state, 0)
// 			    < gsl_vector_get(boundsDir, ids+1)));
//       // Is the state in this box?
//       if (inBox == dim)
// 	{
// 	  foundBox = box;
// 	  break;
// 	}
//     }
  
//   return foundBox;
// }


/** 
 * Get position of grid box.
 * \param[in]  index    Index of box for which to get the position.
 * \param[out] position Position of the box.
 */
void
RegularGrid::getBoxPosition(const size_t index, gsl_vector *position) const
{
  double component;
  gsl_vector *boundsDir;

  for (size_t d = 0; d < dim; d++)
    {
      boundsDir = bounds->at(d);
      component = (gsl_vector_get(boundsDir, index + 1)
		   + gsl_vector_get(boundsDir, index)) / 2;
      gsl_vector_set(position, d, component);
    }

  return;
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
  gridMem = gsl_matrix_uint_alloc(grid->getN(), 2);
  
  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    size_t box0, boxf;
    
#pragma omp for
    for (size_t traj = 0; traj < nTraj; traj++)
      {
	// Find initial box
	gsl_matrix_get_row(X, initStates, traj);
	box0 = grid->getBoxMembership(X);
	
	// Find final box
	gsl_matrix_get_row(X, finalStates, traj);
	boxf = grid->getBoxMembership(X);
	
	// Add transition
#pragma omp critical
	{
	  gsl_matrix_uint_set(gridMem, traj, 0, box0);
	  gsl_matrix_uint_set(gridMem, traj, 1, boxf);
	}
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
    size_t box;
    
#pragma omp for
    for (size_t traj = 0; traj < nStates; traj++)
      {
	// Find box
	gsl_matrix_get_row(X, states, traj);
	box = grid->getBoxMembership(X);

	// Add box
#pragma omp critical
	{
	  gsl_vector_uint_set(gridMem, traj, box);
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
memVector2memMatrix(const gsl_vector_uint *gridMemVect, size_t tauStep)
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
 * \param[in] memList    STD vector of membership Vectors each of them associated
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
 * Get Maximum Likelihood Estimator of a density
 * from the membership vector of a time series to a grid.
 * \param[in]  gridMemVect Grid membership vector.
 * \param[in]  grid        Grid.
 * \param[out] density     MLE of the density on the grid.
 */
void
getDensityMLE(const gsl_vector_uint *gridMemVect, const Grid *grid,
	      gsl_vector *density)
{
  const size_t nt = gridMemVect->size;
  size_t box;

  // Set elements to zero
  gsl_vector_set_zero(density);

  // Add each realization to its box
  for (size_t t = 0; t < nt; t++)
    {
      // Get box index of realization t
      box = gsl_vector_uint_get(gridMemVect, t);

      // Add count to box
      gsl_vector_set(density, box, gsl_vector_get(density, box) + 1);
    }

  // Normalize by number of realizations to get density
  gsl_vector_scale(density, 1. / nt);

  return;
}

// /**
//  * Construct a polar grid adapted to the time series (yet, only for 2D).
//  * \param[in]     nLevels_  Number of levels for the grid
//  * \param[in]     nSectors_ Number of sectors for the grid
//  * \param[in]     support   Rectangular grid used to calculate the density.
//  * \param[in]     states    Trajectory to which to adapt the grid.
//  */
// PolarGrid::PolarGrid(const size_t nLevels_, const size_t nSectors_,
// 		     const Grid *support, const gsl_matrix *states)
//   : Grid(2, nLevels * nSectors), nLevels(nLevels_), nSectors(nSectors_),
//     supportGrid(support)
// {
//   const size_t nStates = states->size1;
//   gsl_vector_uint *statesSupportMem;
//   size_t box, pol;

//   // Allocate
//   allocate(nStates);

//   // Get membership to rectangular grid
//   statesSupportMem = getGridMemVector(states, supportGrid);

//   // Get density
//   getDensityMLE(statesSupportMem, supportGrid, supportDensity);

//   // Get membership of boxes to levels and sectors
//   getSupportMembership(states);

//   // Get membership of realizations to polar grid
//   for (size_t s = 0; s < nStates; s++)
//     {
//       // Get box to which state s belongs
//       box = gsl_vector_uint_get(statesSupportMem, s);
//       if (box == supportGrid->getN())
// 	{
// 	  /** The state is outside the support grid,
// 	      mark it outside the polar grid */
// 	  gsl_vector_uint_set(statesPolarMem, s, N);
// 	}
//       else
// 	{
// 	  // Get polar box to which the state's box belong
// 	  pol = gsl_vector_uint_get(supportPolarMem, box);
	  
// 	  // Add state bo polar box
// 	  gsl_vector_uint_set(statesPolarMem, s, pol);
// 	}
//     }
	
//   // Free
//   gsl_vector_uint_free(statesSupportMem);
// }
  

// /**
//  * Allocate memory for polar grid.
//  * \param[in] nStates Number of states.
//  */
// void
// PolarGrid::allocate(const size_t nStates)
// {
//   supportPolarMem = gsl_vector_uint_alloc(supportGrid->getN());
//   statesPolarMem = gsl_vector_uint_alloc(nStates);
//   supportDensity = gsl_vector_alloc(supportGrid->getN());
  
//   return;
// }

// /**
//  * Destructor desallocating memory.
//  */
// PolarGrid::~PolarGrid()
// {
//   gsl_vector_uint_free(supportPolarMem);
//   gsl_vector_uint_free(statesPolarMem);
//   gsl_vector_free(supportDensity);
// }

// /**
//  * Get membership of support grid-boxes to levels and sectors
//  * based on the density on the support grid.
//  * \param[in] states Time series.
//  */
// void
// PolarGrid::getSupportMembership(const gsl_matrix *states)
// {
//   gsl_vector_uint *supportLevelsMem, *supportSectorsMem;
  
//   /** Get levels */
//   supportLevelsMem = getLevelsMembership();

//   /** Get sectors membership */
//   supportSectorsMem = getSectorsMembership(states);

//   /** Get complete membership */
//   for (size_t b = 0; b < supportGrid->getN(); b++)
//     {
//       gsl_vector_uint_set(supportPolarMem, b,
// 			  gsl_vector_uint_get(supportSectorsMem, b)
// 			  + nSectors * gsl_vector_uint_get(supportLevelsMem, b));
//     }

//   // Free
//   gsl_vector_uint_free(supportLevelsMem);
//   gsl_vector_uint_free(supportSectorsMem);
  
//   return;
// }


// /**
//  * Get support grid membership to levels from its density.
//  */
// gsl_vector_uint *
// PolarGrid::getLevelsMembership()
// {
//   // Allocate vectors
//   const size_t NSupport = supportGrid->getN();
//   gsl_vector *levelsDensity = gsl_vector_calloc(nLevels);
//   gsl_vector *targetDensity = gsl_vector_alloc(nLevels);
//   gsl_vector_uint *isNotUsed = gsl_vector_uint_alloc(NSupport);
//   size_t idxMaxDenAmongNotUsed;
//   size_t nUsed = 0;
//   gsl_vector_uint *supportLevelsMem = gsl_vector_uint_calloc(NSupport);

//   // Set vectors
//   gsl_vector_set_all(targetDensity, 1. / nLevels);
//   gsl_vector_uint_set_all(isNotUsed, 1);

//   // Get levels membership
//   for (size_t lev = 0; lev < nLevels; lev++)
//     {
//       while ((gsl_vector_get(levelsDensity, lev)
// 	      < gsl_vector_get(targetDensity, lev))
// 	     && (nUsed < NSupport))
// 	{
// 	  // Get nonused box with maximum density
// 	  idxMaxDenAmongNotUsed						\
// 	    = gsl_vector_max_index_among(supportDensity, isNotUsed);
	  
// 	  // Add this box to level
// 	  gsl_vector_uint_set(supportLevelsMem, idxMaxDenAmongNotUsed, lev);

// 	  // Add the density of this box to the level
// 	  gsl_vector_set(levelsDensity, lev, gsl_vector_get(levelsDensity, lev)
// 			 + gsl_vector_get(supportDensity, idxMaxDenAmongNotUsed));

// 	  // Mark the box as used
// 	  gsl_vector_uint_set(isNotUsed, idxMaxDenAmongNotUsed, 0);

// 	  // Increment the number of used boxes
// 	  nUsed++;
// 	}
//     }

//   // Free
//   gsl_vector_free(targetDensity);
//   gsl_vector_free(levelsDensity);
//   gsl_vector_uint_free(isNotUsed);
  
//   return supportLevelsMem;
// }


// /**
//  * Get support grid membership to levels.
//  * \param[in] states Time series.
//  */
// gsl_vector_uint *
// PolarGrid::getSectorsMembership(const gsl_matrix *states)
// {
//   const size_t NSupport = supportGrid->getN();
//   size_t idxMode, sect;
//   gsl_vector *modePosition = gsl_vector_alloc(dim);
//   gsl_vector *boxPosition = gsl_vector_alloc(dim);
//   double deltaAngle, boxAngle, EOFAngle;
//   gsl_vector_uint *supportSectorsMem = gsl_vector_uint_alloc(NSupport);
//   gsl_vector *w;
//   gsl_matrix *A, *E;

//   /** Get mode box position */
//   idxMode = gsl_vector_max_index(supportDensity);
//   supportGrid->getBoxPosition(idxMode, modePosition);

//   /** Get first EOF from which to start sectors */
//   w = gsl_vector_alloc(dim);
//   A = gsl_matrix_alloc(states->size1, dim);
//   E = gsl_matrix_alloc(dim, dim);
//   getEOF(states, w, E, A);
//   EOFAngle = atan(gsl_matrix_get(E, 1, 0) / gsl_matrix_get(E, 0, 0));

//   /** Get sector bounds */
//   deltaAngle = 2*M_PI / nSectors;

//   /** Get membership of each support grid-box to a sector */
//   for (size_t b = 0; b < NSupport; b++)
//     {
//       // Get box position
//       supportGrid->getBoxPosition(b, boxPosition);
      
//       // Get box angle
//       boxAngle = atan((gsl_vector_get(boxPosition, 1)
// 		       - gsl_vector_get(modePosition, 1))
// 		      / (gsl_vector_get(boxPosition, 0)
// 			 - gsl_vector_get(modePosition, 0)));

//       // Assign box to sector
//       gsl_vector_uint_set(supportSectorsMem, b,
// 			  fmod(boxAngle - EOFAngle, deltaAngle));
//     }

//   /** Free */
//   gsl_vector_free(modePosition);  
//   gsl_vector_free(boxPosition);
//   gsl_vector_free(w);
//   gsl_matrix_free(A);
//   gsl_matrix_free(E);

//   return supportSectorsMem;
// }

