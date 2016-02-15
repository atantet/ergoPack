#ifndef SSDESOLVERS_HPP
#define SSDESOLVERS_HPP

#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ergoPack/ODESolvers.hpp>
#include <ergoPack/SDESolvers.hpp>


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
  const size_t dim;                   //!< Phase space dimension
  const size_t nDelays;               //!< Number of delayed fields
  const size_t delayMax;              //!< Maximum delay in time steps
  gsl_vector_uint *delays;            //!< Delays
  std::vector<vectorField *> *fields; //!< Vector of delayed vector fields
  gsl_vector *work;                   //!< Workspace used to evaluate the delayed field

public:
  /** \brief Constructor setting the dimension and allocating. */
  vectorFieldDelay(std::vector<vectorField *> *fields_, const gsl_vector_uint *delays_);

  /** \brief Destructor freeing fields and workspace. */
  virtual ~vectorFieldDelay();
  
  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
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
  void updateHistoric(gsl_matrix *currentState);
    
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  numericalSchemeSDDE(const size_t dim_, const size_t nDelays_, const size_t dimWork_,
		      const double dt_)
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
			   gsl_matrix *currentState) = 0;
};


/** \brief Euler-Maruyama SDDE numerical scheme.
 * 
 *  Euler-Maruyama SDDE numerical scheme.
 */
class EulerMaruyamaSDDE : public numericalSchemeSDDE {
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  EulerMaruyamaSDDE(const size_t dim_, const size_t nDelays_, const double dt_)
    : numericalSchemeSDDE(dim_, nDelays_, 2, dt_) {}
  
  /** \brief Destructor freeing workspace. */
  ~EulerMaruyamaSDDE() {}

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  void stepForward(vectorFieldDelay *delayedField, vectorFieldStochastic *stocField,
		   gsl_matrix *currentState);
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
  gsl_matrix *currentState;         //!< Current state (historic)
  
public:
  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting initial historic constant to origin. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_)
    : delayedField(delayedField_),
      dim(delayedField_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  { currentState = gsl_matrix_calloc(delayMax + 1, dim); }
  
  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting a constant initial state. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_, gsl_vector *initStateCst)
    : delayedField(delayedField_),
      dim(delayedField_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  {
    currentState = gsl_matrix_alloc(delayMax + 1, dim);
    for (size_t d = 0; d <= delayMax; d++)
      gsl_matrix_set_row(currentState, d, initStateCst);
  }

  /** \brief Constructor assigning a delayed vector field, a numerical scheme
   *  and a stochastic vector field and setting the initial state. */
  modelSDDE(vectorFieldDelay *delayedField_, vectorFieldStochastic *stocField_,
	    numericalSchemeSDDE *scheme_, gsl_matrix *initState)
    : delayedField(delayedField_),
      dim(delayedField_->getDim()),
      nDelays(delayedField_->getNDelays()),
      delayMax(delayedField_->getDelayMax()),
      stocField(stocField_),
      scheme(scheme_)
  {
    currentState = gsl_matrix_alloc(delayMax + 1, dim);
    gsl_matrix_memcpy(currentState, initState);
  }

  /** \brief Destructor freeing memory. */
  ~modelSDDE() { gsl_matrix_free(currentState); }

  /** \brief One time-step forward integration of the modelSDDE. */
  void stepForward();

  /** \brief Integrate the modelSDDE forward for a given period. */
  gsl_matrix *integrateForward(const double length, const double spinup,
			       const size_t sampling);
};


/*
 * Method definitions
 */

/*
 * Delayed vector field definition 
 */

/**
 * Constructor for a delayed vector field.
 * \param[in] fields_ Vector of vector fields for each delay.
 * \param[in] delays_ Delays associated with each vector field.
 */
vectorFieldDelay::vectorFieldDelay(std::vector<vectorField *> *fields_,
				   const gsl_vector_uint *delays_)
  : dim(fields_->at(0)->getDim()), nDelays(delays_->size),
    delayMax(gsl_vector_uint_get(delays_, nDelays - 1)), fields(fields_)
  {
    // Copy delays
    delays = gsl_vector_uint_alloc(nDelays);
    gsl_vector_uint_memcpy(delays, delays_);
			   
    // Allocate workspace
    work = gsl_vector_alloc(dim);
  }


/**
 * Destructor for a delayed vector field, desallocating.
 */
vectorFieldDelay::~vectorFieldDelay()
{
  for (size_t d = 0; d < nDelays; d++)
    {
      delete fields->at(d);
    }
  delete fields;
  
  gsl_vector_uint_free(delays);
  
  gsl_vector_free(work);
}


/** 
 * Evaluate the delayed vector field from fields for each delay.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
vectorFieldDelay::evalField(gsl_matrix *state, gsl_vector *field)
{
  gsl_vector_view delayedState;
  unsigned int delay;

  // Set field evaluation to 0
  gsl_vector_set_zero(field);

  /** Add delayed drifts */
  for (size_t d = 0; d < nDelays; d++)
    {
      delay = gsl_vector_uint_get(delays, nDelays - d - 1);
      
      // Assign pointer to delayed state
      delayedState = gsl_matrix_row(state, delay);
      
      // Evaluate vector field at delayed state
      fields->at(nDelays - d - 1)->evalField(&delayedState.vector, work);
      
      // Add to newState in workspace
      gsl_vector_add(field, work);
    }

  return;
}


/*
 * Numerical schemes definitions:
 */

/**
 * Update past states of historic by one time step.
 * \param[in,out] currentState Historic to update.
 */
void numericalSchemeSDDE::updateHistoric(gsl_matrix *currentState)
{
  gsl_vector_view delayedState, afterState;
  size_t delayMax = currentState->size1 - 1;

  for (size_t d = 0; d < delayMax; d++)
    {
      delayedState = gsl_matrix_row(currentState, delayMax - d);
      afterState = gsl_matrix_row(currentState, delayMax - d - 1);
      gsl_vector_memcpy(&delayedState.vector, &afterState.vector);
    }

  return;
}


/**
 * Integrate SDDE  one step forward for a given vector field
 * and state using the Euler Maruyama scheme.
 * \param[in]     delayedField Delayed vector fields to evaluate.
 * \param[in]     stocField    stochastic vector field to evaluate.
 * \param[in,out] currentState Current state to update by one time step.
 */
void
EulerMaruyamaSDDE::stepForward(vectorFieldDelay *delayedField,
			       vectorFieldStochastic *stocField,
			       gsl_matrix *currentState)
{
  // Assign pointers to workspace vectors
  gsl_vector_view tmp = gsl_matrix_row(work, 0);
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);
  gsl_vector_view presentState;

  /** Evaluate drift */
  delayedField->evalField(currentState, &tmp.vector);
  // Scale by time step
  gsl_vector_scale(&tmp.vector, dt);

  /** Update historic */
  updateHistoric(currentState);

  // Assign pointer to present state
  presentState = gsl_matrix_row(currentState, 0);

  // Evaluate stochastic field at present state
  stocField->evalField(&presentState.vector, &tmp1.vector); 
  // Scale by time step
  gsl_vector_scale(&tmp1.vector, sqrt(dt));

  // Add drift to present state
  gsl_vector_add(&presentState.vector, &tmp.vector);

  /** Add diffusion at present state */
  gsl_vector_add(&presentState.vector, &tmp1.vector);

  return;
}


/*
 * SDDE model definitions:
 */

/**
 * Integrate one step forward the SDDE model with the numerical scheme.
 */
void
modelSDDE::stepForward()
{
  // Apply numerical scheme to step forward
  scheme->stepForward(delayedField, stocField, currentState);
    
  return;
}


/**
 * Integrate the SDDE model forward for a given period.
 * \param[in]  length   Duration of the integration.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \return              Matrix to record the states.
 */
gsl_matrix *
modelSDDE::integrateForward(const double length, const double spinup,
			const size_t sampling)
{
  size_t nt = length / scheme->getTimeStep();
  size_t ntSpinup = spinup / scheme->getTimeStep();
  gsl_matrix *data = gsl_matrix_alloc((size_t) ((nt - ntSpinup) / sampling), dim);
  gsl_vector_view presentState;

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    {
      // Integrate one step forward
      stepForward();
    }
  
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++)
    {
      // Integrate one step forward
      stepForward();

      // Save present state
      if (i%sampling == 0)
	{
	  presentState = gsl_matrix_row(currentState, 0);
	  gsl_matrix_set_row(data, (i - ntSpinup) / sampling - 1,
			     &presentState.vector);
	}
    }

  return data;
}

/**
 * @}
 */

#endif
