#include <ODESolvers.hpp>
#include <iostream>

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
linearField::evalField(const gsl_vector *state, gsl_vector *field)
{
  // Linear field: apply operator A to state
  gsl_blas_dgemv(CblasNoTrans, 1., A, state, 0., field);

  return;
}


/**
 * Update the matrix of the linear operator
 * conditionned on the  */
void
JacobianQG4::setMatrix(const gsl_vector *x)
{
  // Set non-zero elements
  gsl_matrix_set(A, 0, 0, gsl_vector_get(ci, 0)*gsl_vector_get(x, 1)
		 - gsl_vector_get(li, 0));
  gsl_matrix_set(A, 0, 1, gsl_vector_get(ci, 0)*gsl_vector_get(x, 0)
		 + gsl_vector_get(ci, 1)*gsl_vector_get(x, 2));
  gsl_matrix_set(A, 0, 2, gsl_vector_get(ci, 1)*gsl_vector_get(x, 1)
		 + gsl_vector_get(ci, 2)*gsl_vector_get(x, 3));
  gsl_matrix_set(A, 0, 3, gsl_vector_get(ci, 2) * gsl_vector_get(x, 2));
  gsl_matrix_set(A, 1, 0, gsl_vector_get(ci, 4) * gsl_vector_get(x, 2)
		 - 2 * gsl_vector_get(ci, 0) * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 1, 1, gsl_vector_get(ci, 3)*gsl_vector_get(x, 3)
		 - gsl_vector_get(li, 1));
  gsl_matrix_set(A, 1, 2, gsl_vector_get(ci, 4)*gsl_vector_get(x, 0));
  gsl_matrix_set(A, 1, 3, gsl_vector_get(ci, 3)*gsl_vector_get(x, 1));
  gsl_matrix_set(A, 2, 0, gsl_vector_get(ci, 5)*gsl_vector_get(x, 3)
		 - (gsl_vector_get(ci, 1)+gsl_vector_get(ci, 4))
		 * gsl_vector_get(x, 1));
  gsl_matrix_set(A, 2, 1, -(gsl_vector_get(ci, 1)+gsl_vector_get(ci, 4))
		 * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 2, 2, -gsl_vector_get(li, 2));
  gsl_matrix_set(A, 2, 3, gsl_vector_get(ci, 5)*gsl_vector_get(x, 0));
  gsl_matrix_set(A, 3, 0, -(gsl_vector_get(ci, 2)+gsl_vector_get(ci, 5))
		 * gsl_vector_get(x, 2));
  gsl_matrix_set(A, 3, 1, -2*gsl_vector_get(ci, 3)*gsl_vector_get(x, 1));
  gsl_matrix_set(A, 3, 2, -(gsl_vector_get(ci, 2)+gsl_vector_get(ci, 5))
		 * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 3, 3, -gsl_vector_get(li, 3));

    return;
}

/**
 * Update the matrix of the linear operator
 * conditionned on the  */
void
JacobianQG4Cont::setMatrix(const gsl_vector *x)
{
  // Set last row to 0
  gsl_vector_view view = gsl_matrix_row(A, 4);
  gsl_vector_set_zero(&view.vector);

  // Set non-zero elements
  gsl_matrix_set(A, 0, 0, gsl_vector_get(ci, 0)*gsl_vector_get(x, 1)
		 - gsl_vector_get(li, 0));
  gsl_matrix_set(A, 0, 1, gsl_vector_get(ci, 0)*gsl_vector_get(x, 0)
		 + gsl_vector_get(ci, 1)*gsl_vector_get(x, 2));
  gsl_matrix_set(A, 0, 2, gsl_vector_get(ci, 1)*gsl_vector_get(x, 1)
		 + gsl_vector_get(ci, 2)*gsl_vector_get(x, 3));
  gsl_matrix_set(A, 0, 3, gsl_vector_get(ci, 2) * gsl_vector_get(x, 2));
  gsl_matrix_set(A, 0, 4, 0.);
  gsl_matrix_set(A, 1, 0, gsl_vector_get(ci, 4) * gsl_vector_get(x, 2)
		 - 2 * gsl_vector_get(ci, 0) * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 1, 1, gsl_vector_get(ci, 3)*gsl_vector_get(x, 3)
		 - gsl_vector_get(li, 1));
  gsl_matrix_set(A, 1, 2, gsl_vector_get(ci, 4)*gsl_vector_get(x, 0));
  gsl_matrix_set(A, 1, 3, gsl_vector_get(ci, 3)*gsl_vector_get(x, 1));
  gsl_matrix_set(A, 1, 4, 1.);
  gsl_matrix_set(A, 2, 0, gsl_vector_get(ci, 5)*gsl_vector_get(x, 3)
		 - (gsl_vector_get(ci, 1)+gsl_vector_get(ci, 4))
		 * gsl_vector_get(x, 1));
  gsl_matrix_set(A, 2, 1, -(gsl_vector_get(ci, 1)+gsl_vector_get(ci, 4))
		 * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 2, 2, -gsl_vector_get(li, 2));
  gsl_matrix_set(A, 2, 3, gsl_vector_get(ci, 5)*gsl_vector_get(x, 0));
  gsl_matrix_set(A, 2, 4, 0.);
  gsl_matrix_set(A, 3, 0, -(gsl_vector_get(ci, 2)+gsl_vector_get(ci, 5))
		 * gsl_vector_get(x, 2));
  gsl_matrix_set(A, 3, 1, -2*gsl_vector_get(ci, 3)*gsl_vector_get(x, 1));
  gsl_matrix_set(A, 3, 2, -(gsl_vector_get(ci, 2)+gsl_vector_get(ci, 5))
		 * gsl_vector_get(x, 0));
  gsl_matrix_set(A, 3, 3, -gsl_vector_get(li, 3));
  gsl_matrix_set(A, 3, 4, 0.);

    return;
}

/** 
 * Evaluate the one-dimensional polynomial vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
polynomial1D::evalField(const gsl_vector *state, gsl_vector *field)
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
saddleNodeField::evalField(const gsl_vector *state, gsl_vector *field)
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
transcriticalField::evalField(const gsl_vector *state, gsl_vector *field)
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
pitchforkField::evalField(const gsl_vector *state, gsl_vector *field)
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
pitchforkSubField::evalField(const gsl_vector *state, gsl_vector *field)
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
HopfField::evalField(const gsl_vector *state, gsl_vector *field)
{
  // F(x) = mu*x - x^2
  gsl_vector_set(field, 0, gsl_vector_get(state, 0)
		 * (mu - gsl_vector_get(state, 0)));
  
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
cuspField::evalField(const gsl_vector *state, gsl_vector *field)
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
Lorenz63::evalField(const gsl_vector *state, gsl_vector *field)
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


/** 
 * Evaluate the vector field of the QG4 model
 * at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
QG4::evalField(const gsl_vector *state, gsl_vector *field)
{

  // F1 = c1*A1*A2 + c2*A2*A3 + c3*A3*A4 - l1*A1
  gsl_vector_set(field, 0,
		 gsl_vector_get(ci, 0)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 + gsl_vector_get(ci, 1)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 2)
		 + gsl_vector_get(ci, 2)
		 * gsl_vector_get(state, 2) * gsl_vector_get(state, 3)
		 - gsl_vector_get(li, 0) * gsl_vector_get(state, 0));
  // F2 = c4*A2*A4 + c5*A1*A3 - c1*A1**2 - l2*A2 + sigma
  gsl_vector_set(field, 1,
		 gsl_vector_get(ci, 3)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 3)
		 + gsl_vector_get(ci, 4)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 2)
		 - gsl_vector_get(ci, 0)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 0)
		 - gsl_vector_get(li, 1) * gsl_vector_get(state, 1)
		 + gsl_vector_get(ci, 6) * sigma);
  // F3 = c6*A1*A4 - (c2+c5)*A1*A2 - l3*A3
  gsl_vector_set(field, 2,
		 gsl_vector_get(ci, 5)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 3)
		 - (gsl_vector_get(ci, 1) + gsl_vector_get(ci, 4))
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - gsl_vector_get(li, 2) * gsl_vector_get(state, 2));
  // F4 = -c4*A2**2 - (c3+c6)*A1*A3 - l4*A4
  gsl_vector_set(field, 3,
		 - gsl_vector_get(ci, 3)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 1)
		 - (gsl_vector_get(ci, 2) + gsl_vector_get(ci, 5))
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 2)
		 - gsl_vector_get(li, 3) * gsl_vector_get(state, 3));
 
  return;
}


/** 
 * Evaluate the vector field of the QG4 model
 * at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
QG4Cont::evalField(const gsl_vector *state, gsl_vector *field)
{

  // F1 = c1*A1*A2 + c2*A2*A3 + c3*A3*A4 - l1*A1
  gsl_vector_set(field, 0,
		 gsl_vector_get(ci, 0)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 + gsl_vector_get(ci, 1)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 2)
		 + gsl_vector_get(ci, 2)
		 * gsl_vector_get(state, 2) * gsl_vector_get(state, 3)
		 - gsl_vector_get(li, 0) * gsl_vector_get(state, 0));
  // F2 = c4*A2*A4 + c5*A1*A3 - c1*A1**2 - l2*A2 + sigma
  gsl_vector_set(field, 1,
		 gsl_vector_get(ci, 3)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 3)
		 + gsl_vector_get(ci, 4)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 2)
		 - gsl_vector_get(ci, 0)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 0)
		 - gsl_vector_get(li, 1) * gsl_vector_get(state, 1)
		 + gsl_vector_get(ci, 6) * gsl_vector_get(state, 4));
  // F3 = c6*A1*A4 - (c2+c5)*A1*A2 - l3*A3
  gsl_vector_set(field, 2,
		 gsl_vector_get(ci, 5)
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 3)
		 - (gsl_vector_get(ci, 1) + gsl_vector_get(ci, 4))
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - gsl_vector_get(li, 2) * gsl_vector_get(state, 2));
  // F4 = -c4*A2**2 - (c3+c6)*A1*A3 - l4*A4
  gsl_vector_set(field, 3,
		 - gsl_vector_get(ci, 3)
		 * gsl_vector_get(state, 1) * gsl_vector_get(state, 1)
		 - (gsl_vector_get(ci, 2) + gsl_vector_get(ci, 5))
		 * gsl_vector_get(state, 0) * gsl_vector_get(state, 2)
		 - gsl_vector_get(li, 3) * gsl_vector_get(state, 3));
  // Last element is 0
  gsl_vector_set(field, 4, 0.);
 
  return;
}


/*
 * Numerical schemes definitions:
 */

/**
 * Integrate one step forward for a given vector field and state
 * using the Euler scheme.
 * \param[in]     field   Vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 * \param[in]     dt      Time step.
 */
void
Euler::stepForward(vectorField *field, gsl_vector *current, const double dt)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0); 

  // Get vector field
  field->evalField(current, &tmp.vector);
  
  // Scale by time step
  gsl_vector_scale(&tmp.vector, dt);

  // Add previous state
  gsl_vector_add(current, &tmp.vector);

  return;
}


/**
 * Integrate one step forward for a given vector field and state
 * using the Runge-Kutta 4 scheme.
 * \param[in]     field   Vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 * \param[in]     dt      Time step.
 */
void
RungeKutta4::stepForward(vectorField *field, gsl_vector *current, const double dt)
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
  field->evalField(current, &k1.vector);
  gsl_vector_scale(&k1.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k1.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, current);

  // Second increment
  field->evalField(&tmp.vector, &k2.vector);
  gsl_vector_scale(&k2.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k2.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, current);

  // Third increment
  field->evalField(&tmp.vector, &k3.vector);
  gsl_vector_scale(&k3.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k3.vector);
  gsl_vector_add(&tmp.vector, current);

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
  gsl_vector_add(current, &tmp.vector);

  return;
}


/*
 * Model definitions:
 */

/**
 * Get current state.
 * param[in]  current_ Vector in which to copy the current state.
 */
void
model::getCurrentState(gsl_vector *current_)
{
  gsl_vector_memcpy(current_, current);
  
  return;
}
    
/**
 * Set current state manually.
 * \param[in] current_ Vector to copy to the current state.
 */
void
model::setCurrentState(const gsl_vector *current_)
{
  gsl_vector_memcpy(current, current_);
  
  return;
}


/**
 * Evaluate the vector field.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] vField Vector resulting from the evaluation of the vector field.
 */
void model::evalField(const gsl_vector *state, gsl_vector *vField)
{
  field->evalField(state, vField);

  return;
}


/**
 * Integrate one step forward the model by calling the numerical scheme.
 * \param[in]  dt Time step.
 */
void
model::stepForward(const double dt)
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, current, dt);
    
  return;
}


/**
 * Integrate the model forward for a number of time steps from the current state.
 * If data different from NULL is provided,
 * the sampled states are recorded in *data.
 * If *data is not of the right size, it is reallocated.
 * \param[in]  nt       Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  ntSpinup Initial integration steps to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrateForward(const size_t nt, const double dt,
			const size_t ntSpinup, const size_t sampling,
			gsl_matrix **data)
{
  size_t dataSize = (size_t) ((nt - ntSpinup) / sampling + 0.1);

  // Check if data is of the right size.
  if (data && (((*data)->size1 != dataSize) || ((*data)->size2 != dim)))
    {
      gsl_matrix_free(*data);
      *data = gsl_matrix_alloc(dataSize, dim);
    }

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    stepForward(dt);

  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++)
    {
      stepForward(dt);

      // Save state
      if ((i%sampling == 0) && data)
	gsl_matrix_set_row(*data, (i - ntSpinup) / sampling - 1, current);
    }

  return;
}


/**
 * Integrate the model forward for a number of time steps from a given inital state.
 * If data different from NULL is provided,
 * the sampled states are recorded in *data.
 * If *data is not of the right size, it is reallocated.
 * \param[in]  init     Initial state.
 * \param[in]  nt       Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  ntSpinup Initial integration steps to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrateForward(const gsl_vector *init, const size_t nt, const double dt,
			const size_t ntSpinup, const size_t sampling,
			gsl_matrix **data)
{
  // Initialize state
  setCurrentState(init);

  // Integrate forward
  integrateForward(nt, dt, ntSpinup, sampling, data);
  
  return;
}


/*
 * Integrate the model forward for a given period from the current state.
 * \param[in]  length   Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrateForward(const double length, const double dt,
			const double spinup, const size_t sampling,
			gsl_matrix **data)
{
  size_t nt = (size_t) (length / dt + 0.1);
  size_t ntSpinup = (size_t) (spinup / dt + 0.1);

  integrateForward(nt, dt, ntSpinup, sampling, data);

  return;
}


/**
 * Integrate the model forward for a given period from a given initial state.
 * \param[in]  init     Initial state.
 * \param[in]  length   Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrateForward(const gsl_vector *init, const double length, const double dt,
			const double spinup, const size_t sampling,
			gsl_matrix **data)
{
  size_t nt = (size_t) (length / dt + 0.1);
  size_t ntSpinup = (size_t) (spinup / dt + 0.1);

  integrateForward(init, nt, dt, ntSpinup, sampling, data);

  return;
}

/*
 * Linearized Model definitions:
 */

/**
 * Get current state.
 * param[in]  current_ Matrix in which to copy the current state.
 */
void fundamentalMatrixModel::getCurrentState(gsl_matrix *current_)
{
  gsl_matrix_memcpy(current_, current);
  
  return;
}
    
/**
 * Integrate one step forward the linearized model by calling the numerical scheme.
 * First the full model is integrated forward.
 * Then, integrate forward the linearized model.
 * Last, the Jacobian matrix is updated to the new state.
 * \param[in] dt Time step.
 */
void
fundamentalMatrixModel::stepForward(const double dt)
{
  gsl_vector_view col;

  // Integrate the model forward
  mod->stepForward(dt);

  // Integrate the fundamental matrix forward
  for (size_t d = 0; d < dim; d++)
    {
      // Get matrix column
      col = gsl_matrix_column(current, d); 
  
      // Apply numerical scheme to step forward
      scheme->stepForward(Jacobian, &col.vector, dt);
    }
    
  // Update the Jacobian with the full model current state
  Jacobian->setMatrix(mod->current);
  
  return;
}


/**
 * Integrate the fundamentalMatrixModel forward for a given number of time steps
 * and from the current state.
 * \param[in]  nt      Number of time steps to integrate.
 * \param[in]  dt      Time step.
 */
void
fundamentalMatrixModel::integrateForward(const size_t nt, const double dt)
{
  // Get record
  for (size_t i = 1; i <= nt; i++)
      stepForward(dt);

  return;
}


/**
 * Integrate the fundamentalMatrixModel forward for a given number of time steps
 * and from a given initial state.
 * \param[in]  init    Initial state.
 * \param[in]  initMat Initial fundamental matrix.
 * \param[in]  nt      Number of time steps to integrate.
 * \param[in]  dt      Time step.
 */
void
fundamentalMatrixModel::integrateForward(const gsl_vector *init, const gsl_matrix *initMat,
				  const size_t nt, const double dt)
{
  // Initialize state and fundamental matrix
  setCurrentState(init, initMat);

  // Integrate forward
  integrateForward(nt, dt);

  return;
}


/**
 * Integrate the fundamentalMatrixModel forward for a given period
 * and from the current state.
 * \param[in]  length      Duration of the integration.
 * \param[in]  dt          Time step.
 */
void
fundamentalMatrixModel::integrateForward(const double length, const double dt)
{
  size_t nt = (size_t) (length / dt + 0.1);

  // Get record
  integrateForward(nt, dt);

  return;
}

/**
 * Integrate the fundamentalMatrixModel forward for a given period
 * and from a given initial state.
 * \param[in]  init        Initial state.
 * \param[in]  initMatrix  Initial fundamental matrix.
 * \param[in]  length      Duration of the integration.
 * \param[in]  dt          Time step.
 */
void
fundamentalMatrixModel::integrateForward(const gsl_vector *init, const gsl_matrix *initMat,
				  const double length, const double dt)
{
  size_t nt = (size_t) (length / dt + 0.1);

  // Get record
  integrateForward(init, initMat, nt, dt);

  return;
}

/**
 * Set current fundamental matrix manually.
 * \param[in] currentMat Fundamental matrix to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_matrix *currentMat)
{
  gsl_matrix_memcpy(current, currentMat);
  
  return;
}
    
/**
 * Set current state and fundamental matrix manually.
 * \param[in]  current_   Current state to set to.
 * \param[in]  currentMat Current fundamental matrix to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_vector *current_,
					const gsl_matrix *currentMat)
{
  // Set current state of model
  mod->setCurrentState(current_);
  
  // Update Jacobian to that at current state
  Jacobian->setMatrix(mod->current);
  
  // Update current state of the fundamental matrix
  gsl_matrix_memcpy(current, currentMat);
  
  return;
}
    
/**
 * Set current state manually and fundamental matrix to identity.
 * \param[in]  current_ Current state to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_vector *current_)
{
  // Set current state of model
  mod->setCurrentState(current_);
  
  // Update Jacobian to that at current state
  Jacobian->setMatrix(mod->current);
  
  // Update current state of the fundamental matrix
  gsl_matrix_set_identity(current);
  
  return;
}
    
