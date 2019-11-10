#ifndef COUPLEDRO_HPP
#define COUPLEDRO_HPP

#include <map>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ODESolvers.hpp>
#include <ergoParam.hpp>
#define DIM 3

class Chekroun2019 : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  Chekroun2019(const param *p_) : vectorField(p_) { }

  /** \brief Destructor. */
  ~Chekroun2019() { }

  /** \brief Evaluate the vector field of the coupled recharge oscillator
   *  for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};

#endif
