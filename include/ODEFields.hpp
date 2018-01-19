#ifndef ODEFIELDS_HPP
#define ODEFIELDS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ODESolvers.hpp>

/** \addtogroup simulation
 * @{
 */

/** \file ODEFields.hpp
 *  \brief Ordinary differential equation vector fields.
 *   
 *  Ordinary differential equation vector fields of class vectorField.
 */


/*
 * Class declarations:
 */


/** \brief One-dimensional polynomial vector field
 *
 * One-dimensional polynomial vector field
 *  \f$ F(x) = \sum_{k = 0}^{degree} coeff_k x^k \f$
 */
class polynomial1D : public vectorField {

  const size_t degree; //!< Degree of the polynomial (coeff->size - 1)
  gsl_vector *coeff;   //!< Coefficients of the polynomial
  
public:
  /** \brief Construction by allocating vector of coefficients. */
  polynomial1D(const size_t degree_) : degree(degree_), vectorField()
  { coeff = gsl_vector_alloc(degree + 1); }

  /** \brief Construction by copying coefficients of polynomial. */
    polynomial1D(const gsl_vector *coeff_)
    : degree(coeff_->size - 1), vectorField()
  { coeff = gsl_vector_alloc(degree + 1); gsl_vector_memcpy(coeff, coeff_); }

  /** Destructor freeing the matrix. */
  ~polynomial1D(){ gsl_vector_free(coeff); }

  /** \brief Return the coefficients of the polynomial field (should be allocated first). */
  void getParameters(gsl_vector *coeff_) { gsl_vector_memcpy(coeff_, coeff); return; }
  
  /** \brief Set parameters of the model. */
  void setParameters(const gsl_vector *coeff_) { gsl_vector_memcpy(coeff, coeff_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the saddle-node bifurcation.
 *
 *  Vector field for the normal form of the saddle-node bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *           F(x) = \mu - x^2.
 *
 */
class saddleNodeField : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  saddleNodeField(const param *p_) : vectorField(p_) {}

  /** \brief Evaluate the saddle-node vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the transcritical bifurcation.
 *
 *  Vector field for the normal form of the transcritical bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu - x^2.
 *
 */
class transcriticalField : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  transcriticalField(const param *p_) : vectorField(p_) {}

  /** \brief Evaluate the transcritical vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the supercritical pitchfork bifurcation.
 *
 *  Vector field for the normal form of the supercritical pitchfork bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu x - x^2.
 *
 */
class pitchforkField : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  pitchforkField(const param *p_): vectorField(p_) {}

  /** \brief Evaluate the supercritical pitchfork vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the subcritical pitchfork bifurcation.
 *
 *  Vector field for the normal form of the subcritical pitchfork bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu x + x^3.
 *
 */
class pitchforkSubField : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  pitchforkSubField(const param *p_) : vectorField(p_) {}

  /** \brief Evaluate the subcritical pitchfork vector field. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the cusp bifurcation.
 *
 *  Vector field for the normal form of the cusp bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = h + r x - x^3.
 *
 */
class cuspField : public vectorField {
public:
  /** \brief Default constructor, just defining the phase space dimension. */
  cuspField() : vectorField() {}
  
  /** \brief Constructor defining the model parameters. */
  cuspField(const param *p_) : vectorField(p_) {}

  /** \brief Destructor. */
  ~cuspField(){}

  /** \brief Evaluate the cusp vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Vector field for the normal form of the Hopf bifurcation.
 *
 *  Vector field for the normal form of the Hopf bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F_1(x) = (\mu - (x^2 + y^2)) x - (gamma - \beta (x^2 +y^2)) y
 *            F_2(x) = (gamma - \beta (x^2 +y^2)) x + (\mu - (x^2 + y^2)) y
 *
 */
class Hopf : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  Hopf(const param *p_) : vectorField(p_) {}

  /** \brief Destructor. */
  ~Hopf() {}

  /** \brief Evaluate the vector field of the Lorenz 63 model for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);

};


/** \brief Jacobian of the Hopf normal form.
 *
 *  Jacobian the Hopf normal form.
 */
class JacobianHopf : public linearField {
  public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianHopf(const param *p_) : linearField(2, p_) {}

  /** \brief Destructor. */
  ~JacobianHopf() { }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Vector field for the Hopf normal form.
 *
 *  Vector field for the Hopf normal form with respect to \f$\mu\f$.
 */
class HopfCont : public vectorField {
  
public:
  /** \brief Constructor defining the model parameters. */
  HopfCont(const param *p_) : vectorField(p_) {}

  /** \brief Destructor. */
  ~HopfCont() {}

  /** \brief Evaluate the vector field of the Hopf normal form. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Jacobian of the Hopf normal form.
 *
 *  Jacobian the Hopf normal form.
 */
class JacobianHopfCont : public linearField {
  
public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianHopfCont(const param *p_) : linearField(3, p_) {}

  /** \brief Destructor. */
  ~JacobianHopfCont() { }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Vector field for the Lorenz 63 model.
 *
 *  Vector field for the Lorenz 63 model (Lorenz, 1963):
 *
 *  \f$ F_1(x) = \sigma (x_2 - x_1) \f$
 *
 *  \f$ F_2(x) = x_1 (\rho - x_3) - x_2 \f$
 *
 *  \f$ F_3(x) = x_1 x_2 - \beta x_3 \f$.
 */
class Lorenz63 : public vectorField {
public:
  /** \brief Constructor defining the model parameters. */
  Lorenz63(const param *p_) : vectorField(p_) {}

  /** \brief Destructor. */
  ~Lorenz63() {}

  /** \brief Evaluate the vector field of the Lorenz 63 model for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);

};


/** \brief Jacobian of the Lorenz 63 model.
 *
 *  Jacobian the Lorenz 63 model.
 */
class JacobianLorenz63 : public linearField {
  public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63(const param *p_) : linearField(3, p_) {}

  /** \brief Destructor. */
  ~JacobianLorenz63() { }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Vector field for the Lorenz 63 model for continuation.
 *
 *  Vector field for the Lorenz 63 model (Lorenz, 1963) for continuation
 *  with respect to \f$\rho\f$.
 */
class Lorenz63Cont : public vectorField {
  
public:
  /** \brief Constructor defining the model parameters. */
  Lorenz63Cont(const param *p_) : vectorField(p_) {}

  /** \brief Destructor. */
  ~Lorenz63Cont() {}

  /** \brief Evaluate the vector field of the Lorenz 63 model. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Jacobian of the Lorenz 63 model.
 *
 *  Jacobian the Lorenz 63 model.
 */
class JacobianLorenz63Cont : public linearField {
  
public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63Cont(const param *p_) : linearField(4, p_) {}

  /** \brief Destructor. */
  ~JacobianLorenz63Cont() { }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/**
 * @}
 */

#endif
