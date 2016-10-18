#ifndef ERGOGRID_HPP
#define ERGOGRID_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ios>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl_extension.hpp>
#include <ergoStat.hpp>
#if defined (WITH_OMP) && WITH_OMP == 1
#include <omp.h>
#endif



/** \addtogroup grid
 * @{
 */

/** \file ergoGrid.hpp
 *  \brief Various grid definitions used for Galerkin approximation.
 *  
 *  Various grid definitions used for Galerkin approximation.
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
  
protected:
  const size_t dim;                      //!< Number of dimensions
  const size_t N;                        //!< Number of grid boxes

public:
  /** \brief Constructor allocating an empty grid. */
  Grid(const gsl_vector_uint *nx_)
    : dim(nx_->size), N(gsl_vector_uint_get_prod(nx_)) {}

  /** \brief Constructor allocating an empty grid with same dimensions. */
  Grid(const size_t dim_, const size_t inx)
    : dim(dim_), N(gsl_pow_uint(inx, dim)) {}
  
  /** \brief Destructor. */
  virtual ~Grid() {}

  /** \brief Get state space dimension. */
  size_t getDim() const { return dim; }
  
  /** \brief Get number of grid boxes. */
  size_t getN() const { return N; }
  
  /** \brief Print the grid to file. */
  virtual int printGrid(const char *path, const char *dataFormat,
			const bool verbose) const = 0;

  /** \brief Get membership of a state to a box. */
  virtual size_t getBoxMembership(const gsl_vector *state) const = 0;

  /** \brief Get center of position of grid box. */
  virtual void getBoxPosition(const size_t index, gsl_vector *position) const = 0;
};


/** \brief Curvilinear Grid. */
class CurvilinearGrid : public Grid {
public:
  gsl_vector_uint *nx; //!< Number of grid boxes per dimension

  /** \brief Constructor allocating an empty grid. */
  CurvilinearGrid(const gsl_vector_uint *nx_)
    : Grid(nx_)
  {
      nx = gsl_vector_uint_alloc(dim);
      gsl_vector_uint_memcpy(nx, nx_);
  }

  /** \brief Constructor allocating an empty grid with same dimensions. */
  CurvilinearGrid(const size_t dim_, const size_t inx)
    : Grid(dim_, inx)
  {
    nx = gsl_vector_uint_alloc(dim);
    gsl_vector_uint_set_all(nx, inx);
  }

  /** Destructor desallocates memory for the number of boxes per dimension. */
  ~CurvilinearGrid() { gsl_vector_uint_free(nx); }
};


/** \brief Regular grid. */
class RegularGrid : public CurvilinearGrid {
  
  /** \brief Allocate memory for the grid boundaries. */
  void allocateBounds()
  {
    bounds = new std::vector<gsl_vector *>(dim);
    for (size_t d = 0; d < dim; d++)
      (*bounds)[d] = gsl_vector_alloc(gsl_vector_uint_get(nx, d) + 1);
    return;
  }

  /** \brief Get uniform rectangular. */
  void getRegularGrid(const gsl_vector *gridLimitsLow, const gsl_vector *gridLimitsUp);

  /** \brief Get rectangular grid adapted to the time series. */
  void getAdaptedRegularGrid(const gsl_vector *nSTDLow, const gsl_vector *nSTDHigh,
			     const gsl_matrix *states);

public:
  std::vector<gsl_vector *> *bounds; //!< Grid box bounds for each dimension
  
  /** \brief Construct a uniform rectangular grid with different dimensions. */
  RegularGrid(const gsl_vector_uint *nx_, const gsl_vector *gridLimitsLow, const gsl_vector *gridLimitsUp);
  
  /** \brief Construct a uniform rectangular grid with same dimensions. */
  RegularGrid(const size_t dim_, const size_t inx,
	      const double gridLimitsLow, const double gridLimitsUp);
  
  /** \brief Construct a uniform rectangular grid adapted to time series. */
  RegularGrid(const gsl_vector_uint *nx_,
	      const gsl_vector *nSTDLow, const gsl_vector *nSTDHigh,
	      const gsl_matrix *states);

  /** \brief Construct a uniform rectangular grid with same dimensions adapted to time series. */
  RegularGrid(const size_t inx, const double nSTDLow, const double nSTDHigh,
	      const gsl_matrix *states);

  /** \brief Destructor desallocates memory for the grid boundaries. */
  ~RegularGrid()
  {
    for (size_t d = 0; d < dim; d++)
      gsl_vector_free((*bounds)[d]);
    delete bounds;
  }

  /** \brief Print regular grid to file. */
  int printGrid(const char *path, const char *dataFormat,
		const bool verbose) const;

  /** \brief Get membership of a state to a box. */
  size_t getBoxMembership(const gsl_vector *state) const;

  /** \brief Get center of position of grid box. */
  void getBoxPosition(const size_t index, gsl_vector *position) const;
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
/** \brief Get MLE of a density from the grid membership vector of a time series. */
void getDensityMLE(const gsl_vector_uint *gridMemVect, const Grid *grid,
		   gsl_vector *density);


// /** \brief Polar grid. */
// class PolarGrid : public Grid {
  
//   const size_t nLevels;             //!< Number of levels.
//   const size_t nSectors;            //!< Number of sectors.
//   const Grid *supportGrid;          //!< Support grid used to get the density.
//   gsl_vector_uint *supportPolarMem; //!< Membership of the support grid-boxes to polar grid.
//   gsl_vector_uint *statesPolarMem;  //!< Membership of the support grid-boxes to polar grid.
//   gsl_vector *supportDensity;       //!< Density of states on the support grid.

//   /** \brief Allocate memory for polar grid. */
//   void allocate(const size_t nStates);

//   /** \brief Get membership of support grid-boxes to levels and sectors. */
//   void getSupportMembership(const gsl_matrix *states);

//   /** \brief Get support grid membership to levels from its density. */
//   gsl_vector_uint *getLevelsMembership();

//   /** \brief Get support grid membership to levels. */
//   gsl_vector_uint *getSectorsMembership(const gsl_matrix *states);

// public:
//   /** \brief Construct a polar grid adapted to the time series. */
//   PolarGrid(const size_t nLevels, const size_t nSectors,
// 	    const Grid *support, const gsl_matrix *states);

//   /** \brief Destructor desallocating memory. */
//   ~PolarGrid();

//   /** \brief Print the grid to file. */
//   int printGrid(const char *path, const char *dataFormat,
// 		const bool verbose) const { return 0; }
// };


/**
 * @}
 */

#endif
