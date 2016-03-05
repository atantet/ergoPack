#ifndef SDESOLVERS_HPP
#define SDESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
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

public:
  /** \brief Constructor setting the dimension and allocating. */
  vectorFieldStochastic(const size_t dim_) : vectorField(dim_)
  { noiseState = gsl_vector_alloc(dim); }

  /** \brief Constructor setting the dimension, the generator and allocating. */
  vectorFieldStochastic(const size_t dim_, gsl_rng *rng_)
    : vectorField(dim_), rng(rng_)
  { noiseState = gsl_vector_alloc(dim); }
  
  /** \brief Destructor freeing noise. */
  virtual ~vectorFieldStochastic() { gsl_vector_free(noiseState); }
  
  /** \brief Get noise state. */
  void getNoiseState(gsl_vector *noiseState_)
  { gsl_vector_memcpy(noiseState_, noiseState); return ; }

  /** \brief Update noise realization. */
  void stepForwardNoise() {
    for (size_t i = 0; i < dim; i++)
      gsl_vector_set(noiseState, i, gsl_ran_gaussian(rng, 1.));
  }

  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(gsl_vector *state, gsl_vector *field) = 0;
};


/** \brief Additive Wiener process.
 *
 *  Additive Wiener process stochastic vector field.
 */
class additiveWiener : public vectorFieldStochastic {
  gsl_matrix *Q;  //!< Correlation matrix to apply to noise realization.

public:
  /** \brief Construction by allocating  the matrix of the linear operator. */
  additiveWiener(const size_t dim_) : vectorFieldStochastic(dim_)
  { Q = gsl_matrix_alloc(dim, dim); }
  
  /** \brief Construction by copying the matrix of the linear operator. */
  additiveWiener(const gsl_matrix *Q_, gsl_rng *rng_)
    : vectorFieldStochastic(Q_->size1, rng_)
  { Q = gsl_matrix_alloc(dim, dim); gsl_matrix_memcpy(Q, Q_); }
  
  /** Destructor freeing the matrix. */
  ~additiveWiener(){ gsl_matrix_free(Q); }

  /** \brief Return the parameters of the model (should be allocated first). */
  void getParameters(gsl_matrix *Q_) { gsl_matrix_memcpy(Q_, Q); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *Q_) { gsl_matrix_memcpy(Q, Q_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
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
  void evalField(gsl_vector *state, gsl_vector *field);
};


/** \brief Abstract stochastic numerical scheme class.
 *
 *  Abstract stochastic numerical scheme class.
 */
class numericalSchemeStochastic {
protected:
  const size_t dim;      //!< Dimension of the phase space
  const size_t dimWork;  //!< Dimension of the workspace used to evaluate the field
  double dt;             //!< Time step of integration.
  gsl_matrix *work;      //!< Workspace used to evaluate the vector field
  
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  numericalSchemeStochastic(const size_t dim_, const size_t dimWork_, const double dt_)
    : dim(dim_), dimWork(dimWork_), dt(dt_)
  { work = gsl_matrix_alloc(dimWork, dim); }
  
  /** \brief Destructor freeing workspace. */
  virtual ~numericalSchemeStochastic() { gsl_matrix_free(work); }

  /** \brief Dimension access method. */
  size_t getDim() { return dim; }
  
  /** \brief Return the time step used for the integration. */
  double getTimeStep() { return dt; }

  /** \brief Set or change the time step of integration. */
  void setTimeStep(const double dt_) { dt = dt_; return; }

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  virtual void stepForward(vectorField *field,
			   vectorFieldStochastic *stocField,
			   gsl_vector *currentState) = 0;
};


/** \brief Euler-Maruyama stochastic numerical scheme.
 *  Euler-Maruyama stochastic numerical scheme.
 */
class EulerMaruyama : public numericalSchemeStochastic {
public:
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  EulerMaruyama(const size_t dim_, const double dt_)
    : numericalSchemeStochastic(dim_, 2, dt_) {}
  
  /** \brief Destructor freeing workspace. */
  ~EulerMaruyama() {}

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  void stepForward(vectorField *field, vectorFieldStochastic *stocField,
		   gsl_vector *currentState);
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
class modelStochastic {
  
protected:
  const size_t dim;                  //!< Phase space dimension
  vectorField *field;                //!< Vector field
  vectorFieldStochastic *stocField;  //!< Stochastic vector field
  numericalSchemeStochastic *scheme; //!< Numerical scheme
  gsl_vector *currentState;          //!< Current state
  
public:
  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a stochastic vector field and setting initial state to origin. */
  modelStochastic(vectorField *field_, vectorFieldStochastic *stocField_,
		  numericalSchemeStochastic *scheme_)
    : dim(field_->getDim()), field(field_), stocField(stocField_), scheme(scheme_)
  { currentState = gsl_vector_calloc(dim); }

  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a stochastic vector field and setting initial state. */
  modelStochastic(vectorField *field_, vectorFieldStochastic *stocField_,
		  numericalSchemeStochastic *scheme_, gsl_vector *initState)
    : dim(field_->getDim()), field(field_), stocField(stocField_), scheme(scheme_)
  {
    currentState = gsl_vector_alloc(dim);
    gsl_vector_memcpy(currentState, initState);
  }

  /** \brief Destructor freeing memory. */
  ~modelStochastic() { gsl_vector_free(currentState); }

  /** \brief One time-step forward integration of the modelStochastic. */
  void stepForward();

  /** \brief Integrate the modelStochastic forward for a given period. */
  gsl_matrix *integrateForward(const double length, const double spinup,
			       const size_t sampling);
};


/*
 * Method definitions
 */

void
additiveWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
    
  // Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  return;
}


void
multiplicativeLinearWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
  
  // Additive Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  // Multiply state
  gsl_vector_mul(field, state);

  return;
}


/*
 * Numerical schemes definitions:
 */

/**
 * Integrate stochastically one step forward for a given vector field
 * and state using the Euler Maruyama scheme.
 * \param[in]     field        Vector field to evaluate.
 * \param[in]     stocField    Stochastic vector field to evaluate.
 * \param[in,out] currentState Current state to update by one time step.
 */
void
EulerMaruyama::stepForward(vectorField *field, vectorFieldStochastic *stocField,
			   gsl_vector *currentState)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0);
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);

  // Evalueate fields
  field->evalField(currentState, &tmp.vector);
  stocField->evalField(currentState, &tmp1.vector);

  // Add drift
  gsl_vector_scale(&tmp.vector, dt);
  gsl_vector_add(currentState, &tmp.vector);

  // Add diffusion
  gsl_vector_scale(&tmp1.vector, sqrt(dt));
  gsl_vector_add(currentState, &tmp1.vector);

  return;
}


/**
 * Integrate one step forward the stochastic model with the numerical scheme.
 */
void
modelStochastic::stepForward()
{
  // Apply numerical scheme to step forward
  scheme->stepForward(field, stocField, currentState);
    
  return;
}


/**
 * Integrate the stochastic model forward for a given period.
 * \param[in]  length   Duration of the integration.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \return              Matrix to record the states.
 */
gsl_matrix *
modelStochastic::integrateForward(const double length, const double spinup,
			const size_t sampling)
{
  size_t nt = length / scheme->getTimeStep();
  size_t ntSpinup = spinup / scheme->getTimeStep();
  gsl_matrix *data = gsl_matrix_alloc((size_t) ((nt - ntSpinup) / sampling), dim);

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    stepForward();
  
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++)
    {
      stepForward();

      // Save state
      if (i%sampling == 0)
	gsl_matrix_set_row(data, (i - ntSpinup) / sampling - 1, currentState);
    }

  return data;
}

/**
 * @}
 */

#endif
