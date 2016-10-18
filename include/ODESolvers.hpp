#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

/** \file ODESolvers.hpp
 *  \brief Solve Ordinary Differential Equations.
 *   
 *  Solve Ordinary Differential Equations.
 *  The library uses polymorphism to design a model (class model)
 *  from building blocks.
 *  Those building blocks are the vector field (class vectorField)
 *  and the numerical scheme (class numericalScheme).
 */


/** \ingroup simulation
 * @{
 */

/*
 * Class declarations:
 */

/** \brief Abstract class defining a vector field.
 *
 *  Abstract class defining a vector field. 
 */
class vectorField {
  
protected:
  const size_t dim;   //!< Phase space dimension
  
public:
  /** \brief Constructor setting the dimension. */
  vectorField(const size_t dim_) : dim(dim_) {}
  
  /** \brief Destructor. */
  virtual ~vectorField() {}
  
  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(gsl_vector *state, gsl_vector *field) = 0;
};


/** \brief Vector field defined by a linear operator.
 *
 *  Vector field defined by a linear operator:
 *  \f$ F(x) = A x \f$.
 */
class linearField : public vectorField {
  
gsl_matrix *A;  //!< Matrix representation of the linear operator \f$ A \f$.
  
public:
  /** \brief Construction by allocating matrix of the linear operator. */
  linearField(const size_t dim_) : vectorField(dim_)
  { A = gsl_matrix_alloc(dim, dim); }

  /** \brief Construction by copying the matrix of the linear operator. */
  linearField(const gsl_matrix *A_) : vectorField(A_->size1)
  { A = gsl_matrix_alloc(dim, dim); gsl_matrix_memcpy(A, A_); }

  /** Destructor freeing the matrix. */
  ~linearField(){ gsl_matrix_free(A); }

  /** \brief Return the matrix of the linear operator (should be allocated first). */
  void getParameters(gsl_matrix *A_) { gsl_matrix_memcpy(A_, A); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *A_) { gsl_matrix_memcpy(A, A_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
};


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
  polynomial1D(const size_t degree_) : degree(degree_), vectorField(1)
  { coeff = gsl_vector_alloc(degree + 1); }

  /** \brief Construction by copying coefficients of polynomial. */
    polynomial1D(const gsl_vector *coeff_)
    : degree(coeff_->size - 1), vectorField(1)
  { coeff = gsl_vector_alloc(degree + 1); gsl_vector_memcpy(coeff, coeff_); }

  /** Destructor freeing the matrix. */
  ~polynomial1D(){ gsl_vector_free(coeff); }

  /** \brief Return the coefficients of the polynomial field (should be allocated first). */
  void getParameters(gsl_vector *coeff_) { gsl_vector_memcpy(coeff_, coeff); return; }
  
  /** \brief Set parameters of the model. */
  void setParameters(const gsl_vector *coeff_) { gsl_vector_memcpy(coeff, coeff_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
};


/** \brief Abstract class for codimension one bifurcations of equilibria.
 *
 *  Abstract class for codimension one bifurcations of equilibria
 * (Guckenheimer and Holmes 1988, Strogatz 1994).
 */
class codim1Field : public vectorField {
protected:
  double mu;  //!< Parameter \f$ mu \f$ of the bifurcation.
  
public:
  /** \brief Constructor defining the model parameters. */
  codim1Field(const size_t dim_, const double mu_)
    : vectorField(dim_), mu(mu_) {}

  /** \brief Destructor. */
  ~codim1Field(){}

  /** \brief Return the parameters of the model. */
  virtual void getParameters(double *mu_) { *mu_ = mu; return; }

  /** \brief Set parameters of the model. */
  virtual void setParameters(const double mu_) { mu = mu_; return; }

  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(gsl_vector *state, gsl_vector *field) = 0;
};


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
    : codim1Field(1, mu_) {}

  /** \brief Evaluate the saddle-node vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
    : codim1Field(1, mu_) {}

  /** \brief Evaluate the transcritical vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
    : codim1Field(1, mu_) {}

  /** \brief Evaluate the supercritical pitchfork vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
    : codim1Field(1, mu_) {}

  /** \brief Evaluate the subcritical pitchfork vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
    : codim1Field(2, mu_) {}

  /** \brief Evaluate the Hopf vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
  cuspField() : vectorField(1) {}
  
  /** \brief Constructor defining the model parameters. */
  cuspField(const double r_, const double h_)
    : vectorField(1), r(r_), h(h_) {}

  /** \brief Destructor. */
  ~cuspField(){}

  /** \brief Return the parameters of the model. */
  void getParameters(double *r_, double *h_) { *r_ = r; *h_ = h; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double r_, const double h_) { r = r_; h = h_; return; }

  /** \brief Evaluate the cusp vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
    : vectorField(3),  rho(rho_), sigma(sigma_), beta(beta_) {}

  /** \brief Destructor. */
  ~Lorenz63() {}

  /** \brief Return the parameters of the model. */
  void getParameters(double *rho_, double *sigma_, double *beta_)
  { *rho_ = rho; *sigma_ = sigma; *beta_ = beta; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const double rho_, const double sigma_, const double beta_)
  { rho = rho_; sigma = sigma_; beta = beta_; return; }

  /** \brief Evaluate the vector field of the Lorenz 63 model for a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);

};


/** \brief Abstract defining a deterministic numerical scheme.
 *
 *  Abstract class defining a deterministic numerical scheme used to integrate the model.
 */
class numericalScheme {

protected:
  const size_t dim;      //!< Dimension of the phase space
  const size_t dimWork;  //!< Dimension of the workspace used to evaluate the field
  double dt;             //!< Time step of integration.
  gsl_matrix *work;      //!< Workspace used to evaluate the vector field

public:
  /** \brief Constructor initializing dimensions and allocating. */
  numericalScheme(const size_t dim_, const size_t dimWork_)
    : dim(dim_), dimWork(dimWork_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  numericalScheme(const size_t dim_, const size_t dimWork_, const double dt_)
    : dim(dim_), dimWork(dimWork_), dt(dt_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalScheme() { gsl_matrix_free(work); }

  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
  /** \brief Return the time step used for the integration. */
  double getTimeStep() { return dt; }

  /** \brief Set or change the time step of integration. */
  void setTimeStep(const double dt_) { dt = dt_; return; }

  /** \brief Virtual method to integrate the model one step forward. */
  virtual void stepForward(vectorField *field, gsl_vector *currentState) = 0;

};


/** \brief Euler scheme for numerical integration. 
 * 
 *  Euler scheme for numerical integration. 
 */
class Euler : public numericalScheme {
public:
  /** \brief Constructor defining integration parameters and allocating workspace. */
  Euler(const size_t dim_, const double dt_)
    : numericalScheme(dim_, 1, dt_){}
  
  /** \brief Destructor. */
  ~Euler() {}

  /** \brief One time-step Euler forward Integration of the model. */
  void stepForward(vectorField *field, gsl_vector *currentState);
};


/** \brief Runge-Kutta 4 scheme for numerical integration. 
 * 
 *  Runge-Kutta 4 scheme for numerical integration. 
 */
class RungeKutta4 : public numericalScheme {
public:
  /** \brief Constructor defining integration parameters and allocating workspace. */
  RungeKutta4(const size_t dim_, const double dt_)
    : numericalScheme(dim_, 5, dt_){}
  ~RungeKutta4() {}
  
  /** \brief One time-step Runge-Kutta 4 forward Integration of the model. */
  void stepForward(vectorField *field, gsl_vector *currentState);
};


/** \brief Numerical model class.
 *
 *  Numerical model class.
 *  A model is defined by a vector field and a numerical scheme.
 *  The current state of the model is also recorded.
 *  Attention: the constructors do not copy the vector field
 *  and the numerical scheme given to them, so that
 *  any modification or freeing will affect the model.
 */
class model {
  
protected:
  const size_t dim;                 //!< Phase space dimension
  vectorField *field;               //!< Vector field
  numericalScheme *scheme;          //!< Numerical scheme
  gsl_vector *currentState;         //!< Current state
  
public:
  /** \brief Constructor assigning a vector field and a numerical scheme
   *  and setting initial state to origin. */
  model(vectorField *field_, numericalScheme *scheme_)
    : dim(field_->getDim()), field(field_), scheme(scheme_)
  { currentState = gsl_vector_calloc(dim); }

  /** \brief Constructor assigning a vector field and a numerical scheme
   *  and set initial state. */
  model(vectorField *field_, numericalScheme *scheme_, gsl_vector *initState)
    : dim(field_->getDim()), field(field_), scheme(scheme_)
  {
    currentState = gsl_vector_alloc(dim);
    gsl_vector_memcpy(currentState, initState);
  }

  /** \brief Destructor freeing memory. */
  ~model() { gsl_vector_free(currentState); }

  /** \brief Set current state manually. */
  void setCurrentState(gsl_vector *currentState_)
  {
    gsl_vector_memcpy(currentState, currentState_);
    return;
  }
    
  /** \brief One time-step forward integration of the model. */
  void stepForward();

  /** \brief Integrate the model forward for a given period. */
  gsl_matrix *integrateForward(const double length, const double spinup,
			       const size_t sampling);

};


/**
 * @}
 */

#endif
