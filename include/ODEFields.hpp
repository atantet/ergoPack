#ifndef ODEFIELDS_HPP
#define ODEFIELDS_HPP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ODESolvers.hpp>

/** \file ODEFields.hpp
 *  \brief Ordinary differential equation vector fields.
 *   
 *  Ordinary differential equation vector fields of class vectorField.
 */


/** \ingroup simulation
 * @{
 */

/*
 * Class declarations:
 */


/** \brief Vector field for the normal form of the saddle-node bifurcation.
 *
 *  Vector field for the normal form of the saddle-node bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *           F(x) = \mu - x^2.
 *
 */
class saddleNodeField : public codim1Field {
public:
  /** \brief Constructor defining the model parameters. */
  saddleNodeField(const double mu_)
    : codim1Field(mu_) {}

  /** \brief Evaluate the saddle-node vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the normal form of the transcritical bifurcation.
 *
 *  Vector field for the normal form of the transcritical bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu - x^2.
 *
 */
class transcriticalField : public codim1Field {
public:
  /** \brief Constructor defining the model parameters. */
  transcriticalField(const double mu_)
    : codim1Field(mu_) {}

  /** \brief Evaluate the transcritical vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the normal form of the supercritical pitchfork bifurcation.
 *
 *  Vector field for the normal form of the supercritical pitchfork bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu x - x^2.
 *
 */
class pitchforkField : public codim1Field {
public:
  /** \brief Constructor defining the model parameters. */
  pitchforkField(const double mu_)
    : codim1Field(mu_) {}

  /** \brief Evaluate the supercritical pitchfork vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the normal form of the subcritical pitchfork bifurcation.
 *
 *  Vector field for the normal form of the subcritical pitchfork bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F(x) = \mu x + x^3.
 *
 */
class pitchforkSubField : public codim1Field {
public:
  /** \brief Constructor defining the model parameters. */
  pitchforkSubField(const double mu_)
    : codim1Field(mu_) {}

  /** \brief Evaluate the subcritical pitchfork vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the normal form of the Hopf bifurcation.
 *
 *  Vector field for the normal form of the Hopf bifurcation
 * (Guckenheimer and Holmes, 1988, Strogatz 1994):
 *
 *            F_1(x) = -y + x (\mu - (x^2 + y^2))
 *            F_2(x) = x + y (\mu - (x^2 + y^2)).
 *
 */
class HopfField : public codim1Field {
public:
  /** \brief Constructor defining the model parameters. */
  HopfField(const double mu_)
    : codim1Field(mu_) {}

  /** \brief Evaluate the Hopf vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
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
  
  double r;  //< Parameter \f$ r \f$ related to the pitchfork bifurcation.
  double h;  //< Parameter \f$ h \f$ related to the catastrophe.
  
public:
  /** \brief Default constructor, just defining the phase space dimension. */
  cuspField() : vectorField() {}
  
  /** \brief Constructor defining the model parameters. */
  cuspField(const double r_, const double h_)
    : vectorField(), r(r_), h(h_) {}

  /** \brief Destructor. */
  ~cuspField(){}

  /** \brief Return the parameters of the model. */
  void getParameters(double *r_, double *h_) { *r_ = r; *h_ = h; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double r_, const double h_) { r = r_; h = h_; return; }

  /** \brief Evaluate the cusp vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
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
  
  double rho;     //!< Parameter \f$ \rho \f$ corresponding to the Rayleigh number
  double sigma;   //!< Parameter \f$ \sigma \f$
  double beta;    //!< Parameter \f$ \beta \f$
  
public:
  /** \brief Constructor defining the model parameters. */
  Lorenz63(const double rho_, const double sigma_, const double beta_)
    : vectorField(),  rho(rho_), sigma(sigma_), beta(beta_) {}

  /** \brief Destructor. */
  ~Lorenz63() {}

  /** \brief Return the parameters of the model. */
  void getParameters(double *rho_, double *sigma_, double *beta_)
  { *rho_ = rho; *sigma_ = sigma; *beta_ = beta; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double rho_, const double sigma_, const double beta_)
  { rho = rho_; sigma = sigma_; beta = beta_; return; }

  /** \brief Evaluate the vector field of the Lorenz 63 model for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);

};


/** \brief Jacobian of the Lorenz 63 model.
 *
 *  Jacobian the Lorenz 63 model.
 */
class JacobianLorenz63 : public linearField {
  
  double rho;     //!< Parameter \f$ \rho \f$ corresponding to the Rayleigh number
  double sigma;   //!< Parameter \f$ \sigma \f$
  double beta;    //!< Parameter \f$ \beta \f$
  
public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63(const double rho_, const double sigma_, const double beta_)
    : linearField(3), rho(rho_), sigma(sigma_), beta(beta_) {}

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63(const double rho_, const double sigma_, const double beta_,
		   const gsl_vector *x_)
    : linearField(3), rho(rho_), sigma(sigma_), beta(beta_) {}

  /** \brief Destructor. */
  ~JacobianLorenz63() { }

  /** \brief Return the parameters of the model. */
  void getParameters(double *rho_, double *sigma_, double *beta_)
  { *rho_ = rho; *sigma_ = sigma; *beta_ = beta; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double rho_, const double sigma_, const double beta_)
  { rho = rho_; sigma = sigma_; beta = beta_; return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Jacobian of the Lorenz 63 model.
 *
 *  Jacobian the Lorenz 63 model.
 */
class JacobianLorenz63Cont : public linearField {
  
  double sigma;   //!< Parameter \f$ \sigma \f$
  double beta;    //!< Parameter \f$ \beta \f$
  
public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63Cont(const double sigma_, const double beta_)
    : linearField(4), sigma(sigma_), beta(beta_) {}

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianLorenz63Cont(const double sigma_, const double beta_, const gsl_vector *x_)
    : linearField(4), sigma(sigma_), beta(beta_) {}

  /** \brief Destructor. */
  ~JacobianLorenz63Cont() { }

  /** \brief Return the parameters of the model. */
  void getParameters(double *sigma_, double *beta_)
  { *sigma_ = sigma; *beta_ = beta; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double sigma_, const double beta_)
  { sigma = sigma_; beta = beta_; return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Vector field for the Lorenz 63 model for continuation.
 *
 *  Vector field for the Lorenz 63 model (Lorenz, 1963) for continuation
 *  with respect to \f$\rho\f$.
 */
class Lorenz63Cont : public vectorField {
  
  double sigma;   //!< Parameter \f$ \sigma \f$
  double beta;    //!< Parameter \f$ \beta \f$
  
public:
  /** \brief Constructor defining the model parameters. */
  Lorenz63Cont(const double sigma_, const double beta_)
    : vectorField(), sigma(sigma_), beta(beta_) {}

  /** \brief Destructor. */
  ~Lorenz63Cont() {}

  /** \brief Return the parameters of the model. */
  void getParameters(double *sigma_, double *beta_)
  { *sigma_ = sigma; *beta_ = beta; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double sigma_, const double beta_)
  { sigma = sigma_; beta = beta_; return; }

  /** \brief Evaluate the vector field of the Lorenz 63 model for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);

};


/** \brief Vector field for the quasi-geostrophic 4 modes model.
 *
 *  Vector field for the quasi-geostrophic 4 modes model.
 */
class QG4 : public vectorField {
  double sigma;
  gsl_vector *ci; // c1, ..., c6
  gsl_vector *li; // c1, ..., c6
  
public:
  /** \brief Constructor defining the model parameters. */
  QG4(const double sigma_, const gsl_vector *ci_, const gsl_vector *li_)
    : vectorField(), sigma(sigma_) {
    ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_);
  }

  /** \brief Destructor. */
  ~QG4() { gsl_vector_free(ci); gsl_vector_free(li); }

  /** \brief Return the parameters of the model. */
  void getParameters(double *sigma_, gsl_vector *ci_, gsl_vector *li_)
  { *sigma_ = sigma; gsl_vector_memcpy(ci_, ci); gsl_vector_memcpy(li_, li); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double sigma_, const gsl_vector *ci_, const gsl_vector *li_)
  { sigma = sigma_; gsl_vector_memcpy(ci, ci_); gsl_vector_memcpy(li, li_); return; }

  /** \brief Evaluate the vector field of the quasi-geostrophic 4 modes model
   *  for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the quasi-geostrophic 4 modes model for continuation.
 *
 *  Vector field for the quasi-geostrophic 4 modes model for continuation
 *  with respect to \f$\sigma\f$.
 */
class QG4Cont : public vectorField {
  gsl_vector *ci; // c1, ..., c6
  gsl_vector *li; // mu1, ..., m4
  
public:
  /** \brief Constructor defining the model parameters. */
  QG4Cont(const gsl_vector *ci_, const gsl_vector *li_)
    : vectorField() {
    ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_);
  }

  /** \brief Destructor. */
  ~QG4Cont() { gsl_vector_free(ci); gsl_vector_free(li); }

  /** \brief Return the parameters of the model. */
  void getParameters(gsl_vector *ci_, gsl_vector *li_)
  { gsl_vector_memcpy(ci_, ci); gsl_vector_memcpy(li_, li); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_vector *ci_, const gsl_vector *li_)
  { gsl_vector_memcpy(ci, ci_); gsl_vector_memcpy(li, li_); return; }

  /** \brief Evaluate the vector field of the quasi-geostrophic 4 modes model
   *  for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Jacobian of the quasi-geostrophic 4 modes model.
 *
 *  Jacobian the quasi-geostrophic 4 modes model.
 */
class JacobianQG4 : public linearField {
  
  double sigma;
  gsl_vector *ci;
  gsl_vector *li;

public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianQG4(const double sigma_, const gsl_vector *ci_, const gsl_vector *li_)
    : linearField(4), sigma(sigma_)
  { ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_); }

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianQG4(const double sigma_, const gsl_vector *ci_, const gsl_vector *li_,
	      const gsl_vector *x_) : linearField(4), sigma(sigma_)
  { ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_);
    setMatrix(x_); }

  /** \brief Destructor. */
  ~JacobianQG4() { gsl_vector_free(ci); gsl_vector_free(li); }

  /** \brief Return the parameters of the model. */
  void getParameters(double *sigma_, gsl_vector *ci_, gsl_vector *li_)
  { *sigma_ = sigma; gsl_vector_memcpy(ci_, ci); gsl_vector_memcpy(li, li_);
    return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double sigma_, const gsl_vector *ci_,
		     const gsl_vector *li_)
  { sigma = sigma_; gsl_vector_memcpy(ci, ci_); gsl_vector_memcpy(li, li_); return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Jacobian of the quasi-geostrophic 4 modes model for continuation.
 *
 *  Jacobian of the quasi-geostrophic 4 modes model for continuation
 *  with respect to \f$\sigma\f$.
 */
class JacobianQG4Cont : public linearField {
  
  double sigma;
  gsl_vector * ci;
  gsl_vector * li;

public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianQG4Cont(const gsl_vector *ci_, const gsl_vector *li_)
    : linearField(5)
  { ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_); }

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianQG4Cont(const gsl_vector *ci_, const gsl_vector *li_,
		  const gsl_vector *x_) : linearField(5)
  { ci = gsl_vector_alloc(7); gsl_vector_memcpy(ci, ci_);
    li = gsl_vector_alloc(4); gsl_vector_memcpy(li, li_);
    setMatrix(x_); }

  /** \brief Destructor. */
  ~JacobianQG4Cont() { gsl_vector_free(ci); gsl_vector_free(li); }

  /** \brief Return the parameters of the model. */
  void getParameters(gsl_vector *ci_, gsl_vector *li_)
  { gsl_vector_memcpy(ci_, ci); gsl_vector_memcpy(li, li_);
    return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_vector *ci_, const gsl_vector *li_)
  { gsl_vector_memcpy(ci, ci_); gsl_vector_memcpy(li, li_); return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/**
 * @}
 */

#endif
