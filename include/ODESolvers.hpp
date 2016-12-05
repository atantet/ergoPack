#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

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
  
public:
  /** \brief Constructor. */
  vectorField() {}
  
  /** \brief Destructor. */
  virtual ~vectorField() {}
  
  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(const gsl_vector *state, gsl_vector *field) = 0;

};


/** \brief Vector field defined by a linear operator.
 *
 *  Vector field defined by a linear operator:
 *  \f$ F(x) = A x \f$.
 */
class linearField : public vectorField {
protected:
  gsl_matrix *A;  //!< Matrix representation of the linear operator \f$ A \f$.

public:  
  /** \brief Construction by allocating matrix of the linear operator. */
  linearField(const size_t dim_) : vectorField()
  { A = gsl_matrix_alloc(dim_, dim_); }

  /** \brief Construction by allocating matrix of the linear operator. */
  linearField(const size_t dim1, const size_t dim2) : vectorField()
  { A = gsl_matrix_alloc(dim1, dim2); }

  /** \brief Construction by copying the matrix of the linear operator. */
  linearField(const gsl_matrix *A_) : vectorField()
  { A = gsl_matrix_alloc(A_->size1, A_->size2); gsl_matrix_memcpy(A, A_); }

  /** Destructor freeing the matrix. */
  ~linearField(){ gsl_matrix_free(A); }

  /** \brief Get number of rows of matrix. */
  size_t getRows() { return A->size1; }

  /** \brief Get number of rows of matrix. */
  size_t getCols() { return A->size2; }

  /** \brief Return the matrix of the linear operator (should be allocated first). */
  void getMatrix(gsl_matrix *A_) { gsl_matrix_memcpy(A_, A); return; }

  /** \brief Set parameters of the model. */
  void setMatrix(const gsl_matrix *A_) { gsl_matrix_memcpy(A, A_); return; }

  /** \brief Calculate a new matrix from parameters (do nothing here). */
  virtual void setMatrix(const gsl_vector *x) { };
  
  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
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
  void evalField(const gsl_vector *state, gsl_vector *field);
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
  codim1Field(const double mu_)
    : vectorField(), mu(mu_) {}

  /** \brief Destructor. */
  ~codim1Field(){}

  /** \brief Return the parameters of the model. */
  virtual void getParameters(double *mu_) { *mu_ = mu; return; }

  /** \brief Set parameters of the model. */
  virtual void setParameters(const double mu_) { mu = mu_; return; }

  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(const gsl_vector *state, gsl_vector *field) = 0;
};


/** \brief Abstract defining a deterministic numerical scheme.
 *
 *  Abstract class defining a deterministic numerical scheme used to integrate the model.
 */
class numericalScheme {

protected:
  const size_t dim;      //!< Dimension of the phase space
  const size_t dimWork;  //!< Dimension of the workspace used to evaluate the field
  gsl_matrix *work;      //!< Workspace used to evaluate the vector field

public:
  /** \brief Constructor initializing dimensions and allocating. */
  numericalScheme(const size_t dim_, const size_t dimWork_)
    : dim(dim_), dimWork(dimWork_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalScheme() { gsl_matrix_free(work); }

  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
  /** \brief Virtual method to integrate the model one step forward. */
  virtual void stepForward(vectorField *field, gsl_vector *current,
			   const double dt) = 0;

};


/** \brief Euler scheme for numerical integration. 
 * 
 *  Euler scheme for numerical integration. 
 */
class Euler : public numericalScheme {
public:
  /** \brief Constructor defining integration parameters and allocating workspace. */
  Euler(const size_t dim_) : numericalScheme(dim_, 1){}
  
  /** \brief Destructor. */
  ~Euler() {}

  /** \brief One time-step Euler forward Integration of the model. */
  void stepForward(vectorField *field, gsl_vector *current, const double dt);
};


/** \brief Runge-Kutta 4 scheme for numerical integration. 
 * 
 *  Runge-Kutta 4 scheme for numerical integration. 
 */
class RungeKutta4 : public numericalScheme {
public:
  /** \brief Constructor defining integration parameters and allocating workspace. */
  RungeKutta4(const size_t dim_) : numericalScheme(dim_, 5){}
  ~RungeKutta4() {}
  
  /** \brief One time-step Runge-Kutta 4 forward Integration of the model. */
  void stepForward(vectorField *field, gsl_vector *current, const double dt);
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
  vectorField * const field;               //!< Vector field
  
public:
  numericalScheme * const scheme;          //!< Numerical scheme
  gsl_vector *current;              //!< Current state
  
  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a state. */
  model(vectorField *field_, numericalScheme *scheme_)
    : dim(scheme_->getDim()), field(field_), scheme(scheme_)
  { current = gsl_vector_alloc(dim); }

  /** \brief Destructor freeing memory. */
  ~model() { gsl_vector_free(current); }

  /** \brief Get dimension. */
  size_t getDim(){ return dim; }
    
  /** \brief Get current state. */
  void getCurrentState(gsl_vector *current_);
    
  /** \brief Set current state manually. */
  void setCurrentState(const gsl_vector *current_);

  /** \brief Evaluate the vector field. */
  void evalField(const gsl_vector *state, gsl_vector *vField);

  /** \brief One time-step forward integration of the model. */
  void stepForward(const double dt);

  /** \brief Integrate the model forward for a given number of time steps
   *  from the current state. */
  void integrateForward(const size_t nt, const double dt, const size_t ntSpinup=0,
			const size_t sampling=1, gsl_matrix **data=NULL);
  /** \brief Integrate the model forward for a given number of time steps
   *  from a given initial state. */
  void integrateForward(const gsl_vector *init, const size_t nt, const double dt,
			const size_t ntSpinup=0, const size_t samping=1,
			gsl_matrix **data=NULL);
  /** \brief Integrate the model forward for a given period.
   *  from the current state. */
  void integrateForward(const double length, const double dt, const double spinup=0,
			const size_t sampling=1, gsl_matrix **data=NULL);
  /** \brief Integrate the model forward for a given period.
   *  from a given initial state. */
  void integrateForward(const gsl_vector *init, const double length, const double dt,
			const double spinup=0, const size_t sampling=1,
			gsl_matrix **data=NULL);

};


/** \brief 
 *
 *  Numerical fundamental matrix model.
 *  A fundamentalMatrixModel is defined by a linearized vector field and a numerical scheme.
 *  The current state of the fundamentalMatrixModel is also recorded as a fundamental matrix.
 *  Attention: the constructors do not copy the vector field
 *  and the numerical scheme given to them, so that
 *  any modification or freeing will affect the fundamentalMatrixModel.
 */
class fundamentalMatrixModel {
  
  const size_t dim;                 //!< Phase space dimension
  numericalScheme * const scheme;
  linearField * const Jacobian;            //!< Jacobian Vector field
  
public:
  model * const mod;                       //!< Full model
  gsl_matrix *current;              //!< Current state

  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a state. */
  fundamentalMatrixModel(model *mod_, linearField *Jacobian_)
    : mod(mod_), scheme(mod_->scheme), Jacobian(Jacobian_), dim(mod_->getDim())
  { current = gsl_matrix_alloc(dim, dim); }


  /** \brief Destructor freeing memory. */
  ~fundamentalMatrixModel() { gsl_matrix_free(current); }

  /** \brief Get dimension. */
  size_t getDim(){ return dim; }

  /** \brief Get current state. */
  void getCurrentState(gsl_matrix *current_);
    
  /** \brief Set current fundamental matrix manually. */
  void setCurrentState(const gsl_matrix *currentMat);
    
  /** \brief Set current state and fundamental matrix manually. */
  void setCurrentState(const gsl_vector *current_, const gsl_matrix *currentMat);
    
  /** \brief Set current state manually and fundamental matrix to identity. */
  void setCurrentState(const gsl_vector *current_);
    
  /** \brief One time-step forward integration of the fundamentalMatrixModel. */
  void stepForward(const double dt);

  /** \brief Integrate full and linearized model forward for a number of time steps
   *  from the current state. */
  void integrateForward(const size_t, const double);
  /** \brief Integrate full and linearized model forward for a given period. 
   *  from the current state. */
  void integrateForward(const double, const double);
  /** \brief Integrate full and linearized model forward for a number of time steps
   *  from a given initial state. */
  void integrateForward(const gsl_vector *, const gsl_matrix *,
			const size_t, const double);
  /** \brief Integrate full and linearized model forward for a given period. 
   *  from a given initial state. */
  void integrateForward(const gsl_vector *, const gsl_matrix *,
			const double, const double);

};


/**
 * @}
 */

#endif
