#include <SDESolvers.hpp>

/** \file SDESolvers.cpp
 *  \brief Definitions for SDESolvers.hpp
 *
 */

/*
 * Method definitions
 */

void
additiveWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
    
  // Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  return;
}


void
multiplicativeLinearWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
  
  // Additive Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  // Multiply state
  gsl_vector_mul(field, state);

  return;
}


/*
 * Numerical schemes definitions:
 */

/**
 * Integrate stochastically one step forward for a given vector field
 * and state using the Euler Maruyama scheme.
 * \param[in]     field        Vector field to evaluate.
 * \param[in]     stocField    Stochastic vector field to evaluate.
 * \param[in,out] currentState Current state to update by one time step.
 */
void
EulerMaruyama::stepForward(vectorField *field, vectorFieldStochastic *stocField,
			   gsl_vector *currentState)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0);
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);

  // Evalueate fields
  field->evalField(currentState, &tmp.vector);
  stocField->evalField(currentState, &tmp1.vector);

  // Add drift
  gsl_vector_scale(&tmp.vector, dt);
  gsl_vector_add(currentState, &tmp.vector);

  // Add diffusion
  gsl_vector_scale(&tmp1.vector, sqrt(dt));
  gsl_vector_add(currentState, &tmp1.vector);

  return;
}


/**
 * Integrate one step forward the stochastic model with the numerical scheme.
 */
void
modelStochastic::stepForward()
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, stocField, currentState);
    
  return;
}


/**
 * Integrate the stochastic model forward for a given period.
 * \param[in]  length   Duration of the integration.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \return              Matrix to record the states.
 */
gsl_matrix *
modelStochastic::integrateForward(const double length, const double spinup,
			const size_t sampling)
{
  size_t nt = length / scheme->getTimeStep();
  size_t ntSpinup = spinup / scheme->getTimeStep();
  gsl_matrix *data = gsl_matrix_alloc((size_t) ((nt - ntSpinup) / sampling), dim);

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    stepForward();
  
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++)
    {
      stepForward();

      // Save state
      if (i%sampling == 0)
	gsl_matrix_set_row(data, (i - ntSpinup) / sampling - 1, currentState);
    }

  return data;
}
