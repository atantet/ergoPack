#ifndef SDESOLVERS_HPP
#define SDESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <ODESolvers.hpp>

/** \addtogroup simulation
 * @{
 */

/** \file SDESolvers.hpp
 *  \brief Solve Stocahstic Differential Equations.
 *   
 *  Solve Stochastic Differential Equations.
 *  The library uses polymorphism to design a
 *  stochastic model modelStochastic from building blocks.
 *  These building blocks are the vector field bectorField,
 *  the stochastic vector field stochasticVectorField
 *  and the stochastic numerical scheme stochasticNumericalScheme.
 */


/*
 * Class declarations:
 */

/** \brief Abstract stochastic vector field class.
 * 
 *  Abstract stochastic vector field class inheriting
 *  from the ordinary vector field class.
 */
class vectorFieldStochastic : public vectorField {
  
protected:
  gsl_rng *rng;              //!< Random number gnerator
  gsl_vector *noiseState;    //!< Current noise state (mainly a workspace)
  const size_t noiseDim;     //!< Dimension of Wiener process

public:
  /** \brief Constructor setting the dimension and allocating. */
  vectorFieldStochastic(const size_t noiseDim_)
    : noiseDim(noiseDim_), vectorField()
  { noiseState = gsl_vector_alloc(noiseDim); }

  /** \brief Constructor setting the dim., the generator and allocating. */
  vectorFieldStochastic(const size_t noiseDim_, gsl_rng *rng_)
    : vectorField(), rng(rng_), noiseDim(noiseDim_)
  { noiseState = gsl_vector_alloc(noiseDim); }
  
  /** \brief Constructor setting the dimension and allocating. */
  vectorFieldStochastic(const size_t noiseDim_, const param *p_)
    : noiseDim(noiseDim_), vectorField(p_)
  { noiseState = gsl_vector_alloc(noiseDim); }

  /** \brief Constructor setting the dim., the generator and allocating. */
  vectorFieldStochastic(const size_t noiseDim_, gsl_rng *rng_,
			const param *p_)
    : vectorField(p_), rng(rng_), noiseDim(noiseDim_)
  { noiseState = gsl_vector_alloc(noiseDim); }
  
  /** \brief Destructor freeing noise. */
  virtual ~vectorFieldStochastic() { gsl_vector_free(noiseState); }
  
  /** \brief Get noise state. */
  void getNoiseState(gsl_vector *noiseState_)
  { gsl_vector_memcpy(noiseState_, noiseState); return ; }

  /** \brief Update noise realization. */
  void stepForwardNoise() {
    for (size_t i = 0; i < noiseDim; i++)
      gsl_vector_set(noiseState, i, gsl_ran_gaussian(rng, 1.));
  }

  /** \brief Evaluate the time-dependent stochastic vector field. */
  virtual void evalField(const gsl_vector *state, gsl_vector *field,
			 const double t=0.) = 0;
};


/** \brief Additive Wiener process.
 *
 *  Additive Wiener process stochastic vector field.
 */
class additiveWiener : public vectorFieldStochastic {
  gsl_matrix *Q;  //!< Correlation matrix to apply to noise realization.
  const size_t dim;

public:
  /** \brief Construction by allocating  the matrix of the linear operator. */
  additiveWiener(const size_t dim_, const size_t noiseDim_)
    : vectorFieldStochastic(noiseDim_), dim(dim_)
  { Q = gsl_matrix_alloc(dim, noiseDim); }
  
  /** \brief Construction by copying the matrix of the linear operator. */
  additiveWiener(const gsl_matrix *Q_, gsl_rng *rng_)
    : dim(Q_->size1), vectorFieldStochastic(Q_->size2, rng_)
  { Q = gsl_matrix_alloc(dim, noiseDim); gsl_matrix_memcpy(Q, Q_); }
  
  /** Destructor freeing the matrix. */
  ~additiveWiener(){ gsl_matrix_free(Q); }

  /** \brief Return the parameters of the model (should be allocated first). */
  void getParameters(gsl_matrix *Q_) { gsl_matrix_memcpy(Q_, Q); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *Q_)
  { gsl_matrix_memcpy(Q, Q_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Linear multiplicative Wiener process.
 * 
 *  Linear multiplicative Wiener process stochastic vector field.
 *  Note: nondiagonal state multiplication not (yet) implemented.
 */
class multiplicativeLinearWiener : public vectorFieldStochastic {
  gsl_matrix *Q;   //!< Correlation matrix to apply to noise realization.

public:
  /** \brief Construction by copying the matrix of the linear operator. */
  multiplicativeLinearWiener(const gsl_matrix *Q_, gsl_rng *rng_)
    : vectorFieldStochastic(Q_->size1, rng_)
  { gsl_matrix_memcpy(Q, Q_); }
  
  /** Destructor freeing the matrix. */
  ~multiplicativeLinearWiener(){ gsl_matrix_free(Q); }

  /** \brief Return the parameters of the model. */
  void getParameters(gsl_matrix *Q_) { gsl_matrix_memcpy(Q_, Q); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *Q_) { gsl_matrix_memcpy(Q, Q_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field,
		 const double t=0.);
};


/** \brief Abstract stochastic numerical scheme class.
 *
 *  Abstract stochastic numerical scheme class.
 */
class numericalSchemeStochastic : public numericalSchemeBase {
public:
  /** \brief Constructor initializing dimensions and allocating. */
  numericalSchemeStochastic(const size_t dim_, const size_t dimWork_)
    : numericalSchemeBase(dim_, dimWork_) {}
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalSchemeStochastic() { }

  /** \brief Integrate the model one step. */
  void stepForward(vectorField *field, vectorFieldStochastic *stocField,
		   gsl_vector *current, const double dt, double *t);

  /** \brief Virtual method to integrate the stochastic model one step. */
  virtual gsl_vector_view getStep(vectorField *field,
				  vectorFieldStochastic *stocField,
				  gsl_vector *current, const double dt,
				  const double t) = 0;
};


/** \brief Euler-Maruyama stochastic numerical scheme.
 *  Euler-Maruyama stochastic numerical scheme.
 */
class EulerMaruyama : public numericalSchemeStochastic {
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  EulerMaruyama(const size_t dim_)
    : numericalSchemeStochastic(dim_, 2) {}
  
  /** \brief Destructor freeing workspace. */
  ~EulerMaruyama() {}

  /** \brief Virtual method to get one step of integration. */
  gsl_vector_view getStep(vectorField *field,
			  vectorFieldStochastic *stocField,
			  gsl_vector *current, const double dt,
			  const double t);
};


/** \brief Numerical stochastic model class.
 *
 *  Numerical stochastic model class.
 *  A modelStochastic is defined by a vector field,
 *  a stochastic vector field and a numerical scheme.
 *  The current state of the modelStochastic is also recorded.
 *  Attention: the constructors do not copy the vector field
 *  and the numerical scheme given to them, so that
 *  any modification or freeing will affect the modelStochastic.
 */
class modelStochastic : public modelBase {
  
public:
  vectorFieldStochastic *stocField;  //!< Stochastic vector field
  numericalSchemeStochastic * const scheme;          //!< Numerical scheme
  
  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a stochastic vector field and setting initial state to origin. */
  modelStochastic(vectorField *field_, vectorFieldStochastic *stocField_,
		  numericalSchemeStochastic *scheme_)
    : modelBase(scheme_->getDim(), field_), stocField(stocField_),
      scheme(scheme_) {}

  /** \brief Evaluate the vector field. */
  void evalFieldStochastic(const gsl_vector *state, gsl_vector *vField,
			   const double t);

  /** \brief One time-step forward integration of the modelStochastic. */
  void stepForward(const double dt);
};


/**
 * @}
 */

#endif
