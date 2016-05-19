#include <ODESolvers.hpp>

/** \file ODESolvers.cpp
 *  \brief Definitions for ODESolvers.hpp
 *
 */

/*
 * Method definitions:
 */

/*
 * Vector fields definitions:
 */

/** 
 * Evaluate the linear vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
linearField::evalField(gsl_vector *state, gsl_vector *field)
{
  // Linear field: apply operator A to state
  gsl_blas_dgemv(CblasNoTrans, 1., A, state, 0., field);

  return;
}


/** 
 * Evaluate the one-dimensional polynomial vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
polynomial1D::evalField(gsl_vector *state, gsl_vector *field)
{
  double tmp;
  double stateScal = gsl_vector_get(state, 0);

  // One-dimensional polynomial field:
  tmp = gsl_vector_get(coeff, 0);
  for (size_t c = 1; c < degree + 1; c++)
    {
      tmp += gsl_vector_get(coeff, c) * gsl_pow_uint(stateScal, c);
    }
  
  gsl_vector_set(field, 0, tmp);

  return;
}


/** 
 * Evaluate the vector field of the normal form of the saddleNode bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = \mu - x^2
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
saddleNodeField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu - x^2
  gsl_vector_set(field, 0, mu - pow(gsl_vector_get(state, 0), 2));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the transcritical bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = \mu x - x^2
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
transcriticalField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu*x - x^2
  gsl_vector_set(field, 0, gsl_vector_get(state, 0) * (mu - gsl_vector_get(state, 0)));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the supercritical pitchfork bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = mu x - x^3
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
pitchforkField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu*x - x^3
  gsl_vector_set(field, 0, mu*gsl_vector_get(state, 0)
		 - pow(gsl_vector_get(state, 0), 3));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the subcritical pitchfork bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = mu x + x^3
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
pitchforkSubField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu*x + x^3
  gsl_vector_set(field, 0, mu*gsl_vector_get(state, 0)
		 + pow(gsl_vector_get(state, 0), 3));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the Hopf bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 *
 *            F_1(x) = -y + x (\mu - (x^2 + y^2))
 *            F_2(x) = x + y (\mu - (x^2 + y^2)).
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
HopfField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu*x - x^2
  gsl_vector_set(field, 0, gsl_vector_get(state, 0) * (mu - gsl_vector_get(state, 0)));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the cusp bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = h + r x - x^3
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
cuspField::evalField(gsl_vector *state, gsl_vector *field)
{
  // F(x) = h + r x - x^3
  gsl_vector_set(field, 0, h + r * gsl_vector_get(state, 0) 
		 - pow(gsl_vector_get(state, 0), 3));
  
  return;
}


/** 
 * Evaluate the vector field of the Lorenz 63 model
 * (Lorenz, 1963) at a given state.
 *
 *  \f$ F_1(x) = \sigma (x_2 - x_1) \f$
 *
 *  \f$ F_2(x) = x_1 (\rho - x_3) - x_2 \f$
 *
 *  \f$ F_3(x) = x_1 x_2 - \beta x_3 \f$
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
Lorenz63::evalField(gsl_vector *state, gsl_vector *field)
{

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, sigma
		 * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (rho - gsl_vector_get(state, 2))
		 - gsl_vector_get(state, 1));
  // Fz = x*y - beta*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - beta * gsl_vector_get(state, 2));
 
  return;
}


/*
 * Numerical schemes definitions:
 */

/**
 * Integrate one step forward for a given vector field and state
 * using the Euler scheme.
 * \param[in]     field        Vector field to evaluate.
 * \param[in,out] currentState Current state to update by one time step.
 */
void
Euler::stepForward(vectorField *field, gsl_vector *currentState)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0); 

  // Get vector field
  field->evalField(currentState, &tmp.vector);
  
  // Scale by time step
  gsl_vector_scale(&tmp.vector, dt);

  // Add previous state
  gsl_vector_add(currentState, &tmp.vector);

  return;
}


/**
 * Integrate one step forward for a given vector field and state
 * using the Runge-Kutta 4 scheme.
 * \param[in]     field        Vector field to evaluate.
 * \param[in,out] currentState Current state to update by one time step.
 */
void
RungeKutta4::stepForward(vectorField *field, gsl_vector *currentState)
{
  /** Use views on a working matrix not to allocate memory
   *  at each time step */
  gsl_vector_view k1, k2, k3, k4, tmp; 

  // Assign views
  tmp = gsl_matrix_row(work, 0);
  k1 = gsl_matrix_row(work, 1);
  k2 = gsl_matrix_row(work, 2);
  k3 = gsl_matrix_row(work, 3);
  k4 = gsl_matrix_row(work, 4);
  
  // First increament
  field->evalField(currentState, &k1.vector);
  gsl_vector_scale(&k1.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k1.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, currentState);

  // Second increment
  field->evalField(&tmp.vector, &k2.vector);
  gsl_vector_scale(&k2.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k2.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, currentState);

  // Third increment
  field->evalField(&tmp.vector, &k3.vector);
  gsl_vector_scale(&k3.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k3.vector);
  gsl_vector_add(&tmp.vector, currentState);

  // Fourth increment
  field->evalField(&tmp.vector, &k4.vector);
  gsl_vector_scale(&k4.vector, dt);

  gsl_vector_scale(&k2.vector, 2);
  gsl_vector_scale(&k3.vector, 2);
  gsl_vector_memcpy(&tmp.vector, &k1.vector);
  gsl_vector_add(&tmp.vector, &k2.vector);
  gsl_vector_add(&tmp.vector, &k3.vector);
  gsl_vector_add(&tmp.vector, &k4.vector);
  gsl_vector_scale(&tmp.vector, 1. / 6);

  // Update state
  gsl_vector_add(currentState, &tmp.vector);

  return;
}


/*
 * Model definitions:
 */

/**
 * Integrate one step forward the model by calling the numerical scheme.
 */
void
model::stepForward()
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, currentState);
    
  return;
}


/**
 * Integrate the model forward for a given period.
 * \param[in]  length   Duration of the integration.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \return              Matrix to record the states.
 */
gsl_matrix *
model::integrateForward(const double length, const double spinup,
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

