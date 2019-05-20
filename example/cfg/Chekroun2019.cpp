#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/Chekroun2019.hpp"

void
Chekroun2019::evalField(const gsl_vector *state, gsl_vector *field,
		     const double t)
{
  const double x = gsl_vector_get(state, 0);
  const double y = gsl_vector_get(state, 1);
  const double z = gsl_vector_get(state, 2);
  const double r2 = gsl_pow_2(x) + gsl_pow_2(y);

  //! \f$ F_1(x, y, z) = (\mu - \alpha (x^2 + y^2)) x - (gamma - \beta (x^2 +y^2)) y \f$
  gsl_vector_set(field, 0, (p["mu"] - p["alpha"] * z) * x \
		 - (p["gamma"] + p["delta"] * p["mu"] - p["beta"] * z) * y);
  //! \f$ F_2(x, y, z) = (gamma - \beta (x^2 +y^2)) x + (\mu - \alpha (x^2 + y^2)) y \f$
  gsl_vector_set(field, 1,
		 (p["gamma"] + p["delta"] * p["mu"] - p["beta"] * z) * x \
		 + (p["mu"] - p["alpha"] * z) * y);
  //! \f$ F_3(x, y, z) =  \f$
  gsl_vector_set(field, 2, - (z - r2) / p["sep"]);
 
  return;
}
