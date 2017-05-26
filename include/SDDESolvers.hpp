#ifndef SSDESOLVERS_HPP
#define SSDESOLVERS_HPP

#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ODESolvers.hpp>
#include <SDESolvers.hpp>


/** \addtogroup simulation
 * @{
 */

/** \file SDDESolvers.hpp
 *  \brief Solve Stochastic Delay Differential Equations.
 *   
 *  Solve Stochastic Delay Differential Equations.
 *  The library uses polymorphism to design an
 *  SDDE model modelSDDE from building blocks.
 *  Those building blocks are the delayed vector field vectorFieldDelay,
 *  a stochastic vector field vectorFieldStochastic.
 *  and an SDDE numerical scheme numericalSchemeSDDE.
 */


/*
 * Class declarations:
 */


/** \brief Abstract delayed vector field class.
 * 
 *  Abstract delayed  vector field class.
 */
class vectorFieldDelay {
  
protected:
  const size_t nDelays;               //!< Number of delayed fields
  const size_t delayMax;              //!< Maximum delay in time steps
  gsl_vector_uint *delays;            //!< Delays
  std::vector<vectorField *> *fields; //!< Vector of delayed vector fields
  gsl_vector *work;                   //!< Workspace used to evaluate the delayed field

public:
  /** \brief Constructor setting the dimension and allocating. */
  vectorFieldDelay(std::vector<vectorField *> *fields_,
		   const gsl_vector_uint *delays_);

  /** \brief Destructor freeing fields and workspace. */
  virtual ~vectorFieldDelay();
  
  /** \brief Number of delays access method. */
  size_t getNDelays() { return nDelays; }

  /** \brief Number of delays access method. */
  size_t getDelayMax() { return delayMax; }

  /** \brief Get delays */
  void getDelays(gsl_vector_uint *delays_)
  { gsl_vector_uint_memcpy(delays_, delays); return; }

  /** \brief Method for evaluating the delayed vector field at a given state. */
  void evalField(gsl_matrix *state, gsl_vector *field);
};


/** \brief Abstract SDDE numerical scheme class.
 *
 *  Abstract SDDE numerical scheme class.
 */
class numericalSchemeSDDE {
protected:
  const size_t dim;      //!< Dimension of the phase space
  const size_t nDelays;  //!< Number of delays
  const size_t dimWork;  //!< Dimension of the workspace used to evaluate the field
  double dt;             //!< Time step of integration.
  gsl_matrix *work;      //!< Workspace used to evaluate the vector field

  /** \brief Update past states of a historic by one step */
  void updateHistoric(gsl_matrix *current);
    
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  numericalSchemeSDDE(const size_t dim_, const size_t nDelays_,
		      const size_t dimWork_, const double dt_)
    : dim(dim_), nDelays(nDelays_), dimWork(dimWork_), dt(dt_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalSchemeSDDE() { gsl_matrix_free(work); }

  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
  /** \brief Dimension access method. */
  size_t getNDelays() { return nDelays; }
  
  /** \brief Return the time step used for the integration. */
  double getTimeStep() { return dt; }

  /** \brief Set or change the time step of integration. */
  void setTimeStep(const double dt_) { dt = dt_; return; }

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  virtual void stepForward(vectorFieldDelay *delayedField,
			   vectorFieldStochastic *stocField,
			   gsl_matrix *current) = 0;
};


/** \brief Euler-Maruyama SDDE numerical scheme.
 * 
 *  Euler-Maruyama SDDE numerical scheme.
 */
class EulerMaruyamaSDDE : public numericalSchemeSDDE {
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  EulerMaruyamaSDDE(const size_t dim_, const size_t nDelays_,
		    const double dt_)
    : numericalSchemeSDDE(dim_, nDelays_, 2, dt_) {}
  
  /** \brief Destructor freeing workspace. */
  ~EulerMaruyamaSDDE() {}

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  void stepForward(vectorFieldDelay *delayedField,
		   vectorFieldStochastic *stocField,
		   gsl_matrix *current);
};


/** \brief Numerical SDDE model class.
 *
 *  Numerical SDDE model class.
 *  An SDDE model is defined by a delayed vector field,
 *  a stochastic vector field and a numerical scheme.
 *  The current state (a historic of present and past states)
 *  of the model is also recorded.
 *  Attention: the constructors do not copy the delayed vector field
 *  and the numerical scheme given to them, so that
 *  any modification or freeing will affect the model.
 */
class modelSDDE {
  
protected:
  const size_t dim;                 //!< Phase space dimension
  const size_t nDelays;             //!< Number of delays
  const size_t delayMax;            //!< Maximum delay in number of time steps
  vectorFieldDelay *delayedField;    //!< Vector field for each delay
  vectorFieldStochastic *stocField; //!< Stochastic vector field
  numericalSchemeSDDE *scheme;      //!< Numerical scheme
  gsl_matrix *current;         //!< Current state (historic)
  
public:
  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting initial historic constant to origin. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_)
    : delayedField(delayedField_),
      dim(scheme_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  { current = gsl_matrix_calloc(delayMax + 1, dim); }
  
  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting a constant initial state. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_, gsl_vector *initStateCst)
    : delayedField(delayedField_),
      dim(scheme_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  {
    current = gsl_matrix_alloc(delayMax + 1, dim);
    for (size_t d = 0; d <= delayMax; d++)
      gsl_matrix_set_row(current, d, initStateCst);
  }

  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting the initial state. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_, gsl_matrix *initState)
    : delayedField(delayedField_),
      dim(scheme_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  {
    current = gsl_matrix_alloc(delayMax + 1, dim);
    gsl_matrix_memcpy(current, initState);
  }

  /** \brief Destructor freeing memory. */
  ~modelSDDE() { gsl_matrix_free(current); }

  /** \brief One time-step forward integration of the modelSDDE. */
  void stepForward();

  /** \brief Integrate the modelSDDE forward for a given period. */
  gsl_matrix *integrateForward(const double length, const double spinup,
			       const size_t sampling);
};


/**
 * @}
 */

#endif
