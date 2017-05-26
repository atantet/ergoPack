#include <SDESolvers.hpp>

/** \file SDESolvers.cpp
 *  \brief Definitions for SDESolvers.hpp
 *
 */

/*
 * Method definitions
 */

void
additiveWiener::evalField(const gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
    
  // Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  return;
}


void
multiplicativeLinearWiener::evalField(const gsl_vector *state,
				      gsl_vector *field)
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
 * Integrate one step for a given vector field and state
 * using the scheme.
 * \param[in]     field     Vector field to evaluate.
 * \param[in]     stocField Stochastic vector field to evaluate.
 * \param[in,out] current   Current state to update by one time step.
 * \param[in]     dt        Time step.
 */
void
numericalSchemeStochastic::stepForward(vectorField *field,
				       vectorFieldStochastic *stocField,
				       gsl_vector *current,
				       const double dt)
{
  gsl_vector_view tmp = getStep(field, stocField, current, dt);

  // Add previous state
  gsl_vector_add(current, &tmp.vector);

  return;
}


/**
 * Get one step of integration for a given vector field,
 * stochastic vector field  and state using the Euler-Maruyama scheme.
 * \param[in]     field     Vector field to evaluate.
 * \param[in]     stocField Stochastic vector field to evaluate.
 * \param[in,out] current   Current state to update by one time step.
 * \param[in]     dt        Time step.
 * \return                  A view on the step in the workspace.
 */
gsl_vector_view
EulerMaruyama::getStep(vectorField *field, vectorFieldStochastic *stocField,
		       gsl_vector *current, const double dt)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0); 
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);

  // Get vector field
  field->evalField(current, &tmp.vector);
  stocField->evalField(current, &tmp1.vector);
  
  // Get drift
  gsl_vector_scale(&tmp.vector, dt);

  // Add diffusion
  gsl_vector_scale(&tmp1.vector, sqrt(dt));
  gsl_vector_add(&tmp.vector, &tmp1.vector);

  return tmp;
}


/**
 * Integrate one step forward the stochastic model with the numerical scheme.
 */
void
modelStochastic::stepForward(const double dt)
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, stocField, current, dt);
    
  return;
}


/**
 * Evaluate the stochastic vector field.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] vField Vector resulting from the evaluation of the vector field.
 */
void modelStochastic::evalFieldStochastic(const gsl_vector *state,
					  gsl_vector *vField)
{
  stocField->evalField(state, vField);

  return;
}


