#include <gsl/gsl_math.h>
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
 * Evaluate the one-dimensional polynomial vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
polynomial1D::evalField(const gsl_vector *state, gsl_vector *field,
			const double t)
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
 * \param[in]  t     Time.
 */
void
saddleNodeField::evalField(const gsl_vector *state, gsl_vector *field,
			   const double t)
{
  // F(x) = p["mu"] - x^2
  gsl_vector_set(field, 0, p["mu"] - pow(gsl_vector_get(state, 0), 2));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the transcritical bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = \p["mu"] x - x^2
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
transcriticalField::evalField(const gsl_vector *state, gsl_vector *field,
			      const double t)
{
  // F(x) = p["mu"]*x - x^2
  gsl_vector_set(field, 0, gsl_vector_get(state, 0)
		 * (p["mu"] - gsl_vector_get(state, 0)));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the supercritical pitchfork bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = p["mu"] x - x^3
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
pitchforkField::evalField(const gsl_vector *state, gsl_vector *field,
			  const double t)
{
  // F(x) = p["mu"]*x - x^3
  gsl_vector_set(field, 0, p["mu"] * gsl_vector_get(state, 0)
		 - pow(gsl_vector_get(state, 0), 3));
  
  return;
}


/** 
 * Evaluate the vector field of the normal form of the subcritical pitchfork bifurcation
 * (Guckenheimer & Holmes, 1988, Strogatz,  1994) at a given state:
 * 
 *         F(x) = p["mu"] x + x^3
 *
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
pitchforkSubField::evalField(const gsl_vector *state, gsl_vector *field,
			     const double t)
{
  // F(x) = p["mu"]*x + x^3
  gsl_vector_set(field, 0, p["mu"] * gsl_vector_get(state, 0)
		 + pow(gsl_vector_get(state, 0), 3));
  
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
 * \param[in]  t     Time.
 */
void
cuspField::evalField(const gsl_vector *state, gsl_vector *field,
		     const double t)
{
  // F(x) = h + r x - x^3
  gsl_vector_set(field, 0, p["h"] + p["r"] * gsl_vector_get(state, 0) 
		 - pow(gsl_vector_get(state, 0), 3));
  
  return;
}


/** 
 * Evaluate the vector field of the Hopf normal format a given state.
 * 
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
Hopf::evalField(const gsl_vector *state, gsl_vector *field,
		const double t)
{
  double x, y, r2;
  x = gsl_vector_get(state, 0);
  y = gsl_vector_get(state, 1);
  r2 = gsl_pow_2(x) + gsl_pow_2(y);

  //! \f$ F_1(x) = (\mu - \alpha (x^2 + y^2)) x - (gamma - \beta (x^2 +y^2)) y \f$
  gsl_vector_set(field, 0, (p["mu"] - p["alpha"] * r2) * x \
		 - (p["gamma"] + p["delta"] * p["mu"] - p["beta"] * r2) * y);
  //! \f$ F_2(x) = (gamma - \beta (x^2 +y^2)) x + (\mu - \alpha (x^2 + y^2)) y \f$
  gsl_vector_set(field, 1,
		 (p["gamma"] + p["delta"] * p["mu"] - p["beta"] * r2) * x \
		 + (p["mu"] - p["alpha"] * r2) * y);
 
  return;
}


/**
 * Update the matrix of the Jacobian of the Hopf normal form
 * conditionned on the state.
 * \param[in] state State vector.
*/
void
JacobianHopf::setMatrix(const gsl_vector *state)
{
  double x, y, r2, x2, y2, xy;
  x = gsl_vector_get(state, 0);
  y = gsl_vector_get(state, 1);
  xy = x * y;
  x2 = gsl_pow_2(x);
  y2 = gsl_pow_2(y);
  r2 = x2 + y2;

  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, p["mu"] - r2 - 2*x2 + 2*p["beta"] * xy);
  gsl_matrix_set(A, 0, 1, -2*xy - (p["gamma"] - p["beta"] * r2) \
		 + 2*p["beta"] * y2);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, p["gamma"] - p["beta"] * r2 - 2*p["beta"] * x2 \
		 - 2 * xy);
  gsl_matrix_set(A, 1, 1, -2*p["beta"] * xy + (p["mu"] - r2) - 2*y2);

  return;
}

/** 
 * Evaluate the vector field of the Hopf normal form
 * at a given state for continuation with respect to \f$\rho\f$.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
HopfCont::evalField(const gsl_vector *state, gsl_vector *field,
		    const double t)
{

  double x, y, r2;
  double mu;
  x = gsl_vector_get(state, 0);
  y = gsl_vector_get(state, 1);
  mu = gsl_vector_get(state, 2);
  r2 = gsl_pow_2(x) + gsl_pow_2(y);

  //! \f$ F_1(x) = (\mu - (x^2 + y^2)) x - (gamma - \beta (x^2 +y^2)) y \f$
  gsl_vector_set(field, 0, (mu - r2) * x - (p["gamma"] - p["beta"] * r2) * y);
  //! \f$ F_2(x) = (gamma - \beta (x^2 +y^2)) x + (\mu - (x^2 + y^2)) y \f$
  gsl_vector_set(field, 1,
		 (p["gamma"] - p["beta"] * r2) * x + (mu - r2) * y);
 
  // Last element is 0
  gsl_vector_set(field, 2, 0.);
 
  return;
}


/**
 * Update the matrix of the Jacobian of the Hopf normal form
 * conditionned on the state for continuation with respect to \f$\rho\f$.
 * \param[in] state State vector.
*/
void
JacobianHopfCont::setMatrix(const gsl_vector *state)
{
  double x, y, r2, x2, y2, xy;
  double mu;
  x = gsl_vector_get(state, 0);
  y = gsl_vector_get(state, 1);
  mu = gsl_vector_get(state, 2);
  xy = x * y;
  x2 = gsl_pow_2(x);
  y2 = gsl_pow_2(y);
  r2 = x2 + y2;

  // Set last row to 0
  gsl_vector_view view = gsl_matrix_row(A, 2);
  gsl_vector_set_zero(&view.vector);

  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, mu - r2 - 2*x2 + 2*p["beta"] * xy);
  gsl_matrix_set(A, 0, 1, -2*xy - (p["gamma"] - p["beta"] * r2) \
		 + 2*p["beta"] * y2);
  gsl_matrix_set(A, 0, 2, x);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, p["gamma"] - p["beta"] * r2 - 2*p["beta"] * x2 \
		 - 2 * xy);
  gsl_matrix_set(A, 1, 1, -2*p["beta"] * xy + (mu - r2) - 2*y2);
  gsl_matrix_set(A, 1, 2, y);

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
 * \param[in]  t     Time.
 */
void
Lorenz63::evalField(const gsl_vector *state, gsl_vector *field,
		    const double t)
{

  // Fx = p["sigma"] * (y - x)
  gsl_vector_set(field, 0, p["sigma"]
		 * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (p["rho"] - gsl_vector_get(state, 2))
		 - gsl_vector_get(state, 1));
  // Fz = x*y - p["beta"]*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - p["beta"] * gsl_vector_get(state, 2));
 
  return;
}


/**
 * Update the matrix of the Jacobian of the Lorenz 63 model
 * conditionned on the state.
 * \param[in] state State vector.
*/
void
JacobianLorenz63::setMatrix(const gsl_vector *state)
{
  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, -p["sigma"]);
  gsl_matrix_set(A, 0, 1, p["sigma"]);
  gsl_matrix_set(A, 0, 2, 0.);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, p["rho"] - gsl_vector_get(state, 2));
  gsl_matrix_set(A, 1, 1, -1.);
  gsl_matrix_set(A, 1, 2, -gsl_vector_get(state, 0));
  // Set row for \f$x_3\f$
  gsl_matrix_set(A, 2, 0, gsl_vector_get(state, 1));
  gsl_matrix_set(A, 2, 1, gsl_vector_get(state, 0));
  gsl_matrix_set(A, 2, 2, -p["beta"]);

  return;
}

/** 
 * Evaluate the vector field of the Lorenz 63 model
 * at a given state for continuation with respect to \f$\rho\f$.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 * \param[in]  t     Time.
 */
void
Lorenz63Cont::evalField(const gsl_vector *state, gsl_vector *field,
			const double t)
{

  // Fx = p["sigma"] * (y - x)
  gsl_vector_set(field, 0, p["sigma"]
		 * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (gsl_vector_get(state, 3) - gsl_vector_get(state, 2))
		 - gsl_vector_get(state, 1));
  // Fz = x*y - p["beta"]*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - p["beta"] * gsl_vector_get(state, 2));
  // Last element is 0
  gsl_vector_set(field, 3, 0.);
 
  return;
}


/**
 * Update the matrix of the Jacobian of the Lorenz 63 model
 * conditionned on the state for continuation with respect to \f$\rho\f$.
 * \param[in] state State vector.
*/
void
JacobianLorenz63Cont::setMatrix(const gsl_vector *state)
{
  // Set last row to 0
  gsl_vector_view view = gsl_matrix_row(A, 3);
  gsl_vector_set_zero(&view.vector);

  // Set row for \f$x_1\f$
  gsl_matrix_set(A, 0, 0, -p["sigma"]);
  gsl_matrix_set(A, 0, 1, p["sigma"]);
  gsl_matrix_set(A, 0, 2, 0.);
  gsl_matrix_set(A, 0, 3, 0.);
  // Set row for \f$x_2\f$
  gsl_matrix_set(A, 1, 0, gsl_vector_get(state, 3)
		 - gsl_vector_get(state, 2));
  gsl_matrix_set(A, 1, 1, -1.);
  gsl_matrix_set(A, 1, 2, -gsl_vector_get(state, 0));
  gsl_matrix_set(A, 1, 3, gsl_vector_get(state, 0));
  // Set row for \f$x_3\f$
  gsl_matrix_set(A, 2, 0, gsl_vector_get(state, 1));
  gsl_matrix_set(A, 2, 1, gsl_vector_get(state, 0));
  gsl_matrix_set(A, 2, 2, -p["beta"]);
  gsl_matrix_set(A, 2, 3, 0.);

  return;
}
