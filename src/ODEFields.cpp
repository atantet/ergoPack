#include <ODESolvers.hpp>
#include <ODEFields.hpp>

/** \file ODEFields.cpp
 *  \brief Definitions for Ordinary differential equation vector fields.
 *   
 *  Definitions Ordinary differential equation vector fields of class vectorField.
 */

/*
 * Method definitions:
 */

/*
 * Vector fields definitions:
 */

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
 * Vector fields and Jacobian for the Lorenz 63 model.
 */

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
 * Update the matrix of the Jacobian of the Lorenz 63 model
 * conditionned on the state x.
 * \param[in] x State vector.
*/
void
JacobianLorenz63::setMatrix(const gsl_vector *x)
{
  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, -sigma);
  gsl_matrix_set(A, 0, 1, sigma);
  gsl_matrix_set(A, 0, 2, 0.);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, rho - gsl_vector_get(x, 2));
  gsl_matrix_set(A, 1, 1, -1.);
  gsl_matrix_set(A, 1, 2, -gsl_vector_get(x, 0));
  // Set row for \f$x_3\f$
  gsl_matrix_set(A, 2, 0, gsl_vector_get(x, 1));
  gsl_matrix_set(A, 2, 1, gsl_vector_get(x, 0));
  gsl_matrix_set(A, 2, 2, -beta);

  return;
}

/** 
 * Evaluate the vector field of the Lorenz 63 model
 * at a given state for continuation with respect to \f$\rho\f$.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
Lorenz63Cont::evalField(const gsl_vector *state, gsl_vector *field)
{

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, sigma
		 * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (gsl_vector_get(state, 3) - gsl_vector_get(state, 2))
		 - gsl_vector_get(state, 1));
  // Fz = x*y - beta*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - beta * gsl_vector_get(state, 2));
  // Last element is 0
  gsl_vector_set(field, 4, 0.);
 
  return;
}


/**
 * Update the matrix of the Jacobian of the Lorenz 63 model
 * conditionned on the state x for continuation with respect to \f$\rho\f$.
 * \param[in] x State vector.
*/
void
JacobianLorenz63Cont::setMatrix(const gsl_vector *x)
{
  // Set last row to 0
  gsl_vector_view view = gsl_matrix_row(A, 3);
  gsl_vector_set_zero(&view.vector);

  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, -sigma);
  gsl_matrix_set(A, 0, 1, sigma);
  gsl_matrix_set(A, 0, 2, 0.);
  gsl_matrix_set(A, 0, 3, 0.);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, gsl_vector_get(x, 3) - gsl_vector_get(x, 2));
  gsl_matrix_set(A, 1, 1, -1.);
  gsl_matrix_set(A, 1, 2, -gsl_vector_get(x, 0));
  gsl_matrix_set(A, 1, 3, gsl_vector_get(x, 0));
  // Set row for \f$x_3\f$
  gsl_matrix_set(A, 2, 0, gsl_vector_get(x, 1));
  gsl_matrix_set(A, 2, 1, gsl_vector_get(x, 0));
  gsl_matrix_set(A, 2, 2, -beta);
  gsl_matrix_set(A, 2, 3, 0.);

  return;
}


/**
 * Vector fields and Jacobian for the quasi-geostrophic 4 modes model.
 */

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


/**
 * Update the matrix of the Jacobian of the quasi-geostrophic 4 modes model
 * conditionned on the state x.
 * \param[in] x State vector.
*/
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
 * Update the matrix of the Jacobian of the quasi-geostrophic 4 modes model
 * conditionned on the state x for continuation with respect to \f$\sigma\f$.
 * \param[in] x State vector.
*/
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


