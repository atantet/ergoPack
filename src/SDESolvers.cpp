#include <SDESolvers.hpp>

/** \file SDESolvers.cpp
 *  \brief Definitions for SDESolvers.hpp
 *
 */

/*
 * Method definitions
 */

/** 
 * Evaluate white noise vector field.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time at which to evaluate vector fields.
 */
void
additiveWiener::evalField(const gsl_vector *state, gsl_vector *field,
			  const double t)
{
  // Get new noise realization
  stepForwardNoise();
    
  // Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  return;
}


/** 
 * Evaluate linear multiplicative noise vector field.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time at which to evaluate vector fields.
 */
void
multiplicativeLinearWiener::evalField(const gsl_vector *state,
				      gsl_vector *field, const double t)
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
 * \param[in]     t         Time at which to evaluate the fields.
 */
void
numericalSchemeStochastic::stepForward(vectorField *field,
				       vectorFieldStochastic *stocField,
				       gsl_vector *current,
				       const double dt, double *t)
{
  // Get next step
  gsl_vector_view tmp = getStep(field, stocField, current, dt, *t);

  // Add previous state
  gsl_vector_add(current, &tmp.vector);

  // Update time
  *t += dt;

  return;
}


/**
 * Get one step of integration for a given vector field,
 * stochastic vector field  and state using the Euler-Maruyama scheme.
 * \param[in]     field     Vector field to evaluate.
 * \param[in]     stocField Stochastic vector field to evaluate.
 * \param[in,out] current   Current state to update by one time step.
 * \param[in]     dt        Time step.
 * \param[in]     t         Time at which to evaluate the fields.
 * \return                  A view on the step in the workspace.
 */
gsl_vector_view
EulerMaruyama::getStep(vectorField *field, vectorFieldStochastic *stocField,
		       gsl_vector *current, const double dt, const double t)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0); 
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);

  // Get vector field
  field->evalField(current, &tmp.vector, t);
  stocField->evalField(current, &tmp1.vector, t);
  
  // Get drift
  gsl_vector_scale(&tmp.vector, dt);

  // Add diffusion
  gsl_vector_scale(&tmp1.vector, sqrt(dt));
  gsl_vector_add(&tmp.vector, &tmp1.vector);

  return tmp;
}


/**
 * Integrate one step forward the stochastic model with the numerical scheme.
 * \param[in] dt Time step.
 */
void
modelStochastic::stepForward(const double dt)
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, stocField, current, dt, &t);
    
  return;
}


/**
 * Evaluate the stochastic vector field.
 * \param[in]  state  State at which to evaluate the vector field.
 * \param[out] vField Vector resulting from the evaluation of the vector field.
 * \param[in]  t      Time at which to evaluate the fields.
 */
void modelStochastic::evalFieldStochastic(const gsl_vector *state,
					  gsl_vector *vField, const double t)
{
  stocField->evalField(state, vField, t);

  return;
}
