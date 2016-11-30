#include <SDDESolvers.hpp>

/** \file SDDESolvers.cpp
 *  \brief Definitions for SDDESolvers.hpp
 *
 */

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
 * \param[in,out] current Historic to update.
 */
void numericalSchemeSDDE::updateHistoric(gsl_matrix *current)
{
  gsl_vector_view delayedState, afterState;
  size_t delayMax = current->size1 - 1;

  for (size_t d = 0; d < delayMax; d++)
    {
      delayedState = gsl_matrix_row(current, delayMax - d);
      afterState = gsl_matrix_row(current, delayMax - d - 1);
      gsl_vector_memcpy(&delayedState.vector, &afterState.vector);
    }

  return;
}


/**
 * Integrate SDDE  one step forward for a given vector field
 * and state using the Euler Maruyama scheme.
 * \param[in]     delayedField Delayed vector fields to evaluate.
 * \param[in]     stocField    stochastic vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 */
void
EulerMaruyamaSDDE::stepForward(vectorFieldDelay *delayedField,
			       vectorFieldStochastic *stocField,
			       gsl_matrix *current)
{
  // Assign pointers to workspace vectors
  gsl_vector_view tmp = gsl_matrix_row(work, 0);
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);
  gsl_vector_view presentState;

  /** Evaluate drift */
  delayedField->evalField(current, &tmp.vector);
  // Scale by time step
  gsl_vector_scale(&tmp.vector, dt);

  /** Update historic */
  updateHistoric(current);

  // Assign pointer to present state
  presentState = gsl_matrix_row(current, 0);

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
  scheme->stepForward(delayedField, stocField, current);
    
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
	  presentState = gsl_matrix_row(current, 0);
	  gsl_matrix_set_row(data, (i - ntSpinup) / sampling - 1,
			     &presentState.vector);
	}
    }

  return data;
}

