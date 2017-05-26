#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <vector>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ergoParam.hpp>

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
  param p; //!< Parameters.
  
public:
  /** \brief Default onstructor. */
  vectorField() {}
  
  /** \brief Constructor. */
  vectorField(const param *p_) : p(*p_) {}
  
  /** \brief Destructor. */
  virtual ~vectorField() {}
  
  /** \brief Return the parameters of the model. */
  virtual void getParameters(param *p_) { *p_ = p; return; }

  /** \brief Set parameters of the model. */
  virtual void setParameters(const param *p_) { p = *p_; return; }

  /** \brief Evaluate the vector field at a given state. */
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
  linearField(const size_t dim_, const param *p_) : vectorField(p_)
  { A = gsl_matrix_alloc(dim_, dim_); }

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


/** \brief Abstract base class defining a deterministic numerical scheme.
 *
 *  Abstract base class defining a deterministic numerical scheme
 *  used to integrate the model.
 */
class numericalSchemeBase {

protected:
  const size_t dim;      //!< Dimension of the phase space
  const size_t dimWork;  //!< Dimension of the workspace used to evaluate the field
  gsl_matrix *work;      //!< Workspace used to evaluate the vector field

public:
  /** \brief Constructor initializing dimensions and allocating. */
  numericalSchemeBase(const size_t dim_, const size_t dimWork_)
    : dim(dim_), dimWork(dimWork_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalSchemeBase() { gsl_matrix_free(work); }

  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
};


/** \brief Abstract defining a deterministic numerical scheme.
 *
 *  Abstract class defining a deterministic numerical scheme
 *  used to integrate the model.
 */
class numericalScheme : public numericalSchemeBase {

public:
  /** \brief Constructor initializing dimensions and allocating. */
  numericalScheme(const size_t dim_, const size_t dimWork_)
    : numericalSchemeBase(dim_, dimWork_) {}
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalScheme() { }

  /** \brief Integrate the model one step. */
  void stepForward(vectorField *field, gsl_vector *current,
		   const double dt);

  /** \brief Virtual method to get one step of integration. */
  virtual gsl_vector_view getStep(vectorField *field, gsl_vector *current,
				  const double dt) = 0;
};


/** \brief Euler scheme for numerical integration. 
 * 
 *  Euler scheme for numerical integration. 
 */
class Euler : public numericalScheme {
public:
  /** \brief Constructor. Define integration param. and allocate workspace. */
  Euler(const size_t dim_) : numericalScheme(dim_, 1){}
  
  /** \brief Destructor. */
  ~Euler() {}

  /** \brief Virtual method to get one step of integration. */
  gsl_vector_view getStep(vectorField *field, gsl_vector *current,
			  const double dt);
};


/** \brief Runge-Kutta 4 scheme for numerical integration. 
 * 
 *  Runge-Kutta 4 scheme for numerical integration. 
 */
class RungeKutta4 : public numericalScheme {
public:
  /** \brief Constructor. Define integration param. and allocate workspace. */
  RungeKutta4(const size_t dim_) : numericalScheme(dim_, 5){}

  /** \brief Destructor. */
  ~RungeKutta4() {}
  
  /** \brief Virtual method to get one step of integration. */
  gsl_vector_view getStep(vectorField *field, gsl_vector *current,
			  const double dt);
};


/** \brief Numerical model base class.
 *
 *  Numerical model base class.
 */
class modelBase {
  
protected:
  const size_t dim;                 //!< Phase space dimension
  
public:
  vectorField * const field;               //!< Vector field
  gsl_vector *current;              //!< Current state
  
  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a state. */
  modelBase(const size_t dim_, vectorField *field_) : dim(dim_), field(field_)
  { current = gsl_vector_alloc(dim); }

  /** \brief Destructor freeing memory. */
  virtual ~modelBase() { gsl_vector_free(current); }

  /** \brief Get dimension. */
  size_t getDim(){ return dim; }
    
  /** \brief Get current state. */
  void getCurrentState(gsl_vector *current_);
    
  /** \brief Set current state manually. */
  void setCurrentState(const gsl_vector *current_);

  /** \brief Evaluate the vector field. */
  void evalField(const gsl_vector *state, gsl_vector *vField);

  /** \brief One time-step integration of the model. */
  virtual void stepForward(const double dt) = 0;

  /** \brief Integrate the model for a given number of time steps
   *  from the current state. */
  void integrate(const size_t nt, const double dt, const size_t ntSpinup=0,
		 const size_t sampling=1, gsl_matrix **data=NULL);
  /** \brief Integrate the model for a given number of time steps
   *  from a given initial state. */
  void integrate(const gsl_vector *init, const size_t nt, const double dt,
		 const size_t ntSpinup=0, const size_t samping=1,
		 gsl_matrix **data=NULL);
  /** \brief Integrate the model for a given period.
   *  from the current state. */
  void integrate(const double length, const double dt, const double spinup=0,
		 const size_t sampling=1, gsl_matrix **data=NULL);
  /** \brief Integrate the model for a given period.
   *  from a given initial state. */
  void integrate(const gsl_vector *init, const double length,
		 const double dt, const double spinup=0,
		 const size_t sampling=1, gsl_matrix **data=NULL);

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
class model : public modelBase {
  
public:
  numericalScheme * const scheme;          //!< Numerical scheme
  
  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a state. */
  model(vectorField *field_, numericalScheme *scheme_)
    : modelBase(scheme_->getDim(), field_), scheme(scheme_) {}

  /** \brief Destructor freeing memory. */
  ~model() { }

  /** \brief One time-step integration of the model. */
  virtual void stepForward(const double dt);
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

  /** \brief Constructor assigning a vector field, a numerical scheme. */
  fundamentalMatrixModel(model *mod_, linearField *Jacobian_)
    : mod(mod_), scheme(mod_->scheme), Jacobian(Jacobian_),
      dim(mod_->getDim())
  { current = gsl_matrix_alloc(dim, dim); }

  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a state. */
  fundamentalMatrixModel(model *mod_, linearField *Jacobian_,
			 const gsl_vector *initState)
    : mod(mod_), scheme(mod_->scheme), Jacobian(Jacobian_),
      dim(mod_->getDim())
  { current = gsl_matrix_alloc(dim, dim);
    setCurrentState(initState); }


  /** \brief Destructor freeing memory. */
  ~fundamentalMatrixModel() { gsl_matrix_free(current); }

  /** \brief Get dimension. */
  size_t getDim(){ return dim; }

  /** \brief Get current state. */
  void getCurrentState(gsl_matrix *current_);
    
  /** \brief Set current fundamental matrix manually. */
  void setCurrentState(const gsl_matrix *currentMat);
    
  /** \brief Set current state and fundamental matrix manually. */
  void setCurrentState(const gsl_vector *current_,
		       const gsl_matrix *currentMat);
    
  /** \brief Set current state manually and fundamental matrix to identity. */
  void setCurrentState(const gsl_vector *current_);
    
  /** \brief Update Jacobian to current model state
   * and set fundamental matrix to identity. */
  void setCurrentState();

  /** \brief One time-step integration offundamentalMatrixModel. */
  void stepForward(const double dt);

  /** \brief Integrate full and linear model for a number of time steps
   *  from the current state. */
  void integrate(const size_t nt, const double dt);
  
  /** \brief Integrate full and linearized model for a given period. 
   *  from the current state. */
  void integrate(const double length, const double dt);
  
  /** \brief Integrate full and linear model for a number of time steps
   *  from a given initial state. */
  void integrate(const gsl_vector *, const size_t nt, const double dt);
  
  /** \brief Integrate full and linearized model for a given period. 
   *  from a given initial state. */
  void integrate(const gsl_vector *init, const double length,
		 const double dt);

  /** \brief Get the fundamental matrices Mts for s between 0 and t. */
  void integrateRange(const size_t nt, const double dt, gsl_matrix **xt,
		      std::vector<gsl_matrix *> *Mts,
		      const size_t ntSpinup=0);

  /** \brief Get the fund. mat. Mts for s between 0 and t with init. */
  void integrateRange(const gsl_vector *init, const size_t nt,
		      const double dt, gsl_matrix **xt,
		      std::vector<gsl_matrix *> *Mts,
		      const size_t ntSpinup=0);
};


/**
 * @}
 */

#endif
