#include <vector>
#include <ODESolvers.hpp>
#include <gsl/gsl_blas.h>

/** \file ODESolvers.cpp
 *  \brief Definitions for ODESolvers.hpp
 *
 */

/*
 * Method definitions:
 */

/*
 * Vector fields definitions:
 */

/** 
 * Evaluate the linear vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
linearField::evalField(const gsl_vector *state, gsl_vector *field)
{
  // Linear field: apply operator A to state
  gsl_blas_dgemv(CblasNoTrans, 1., A, state, 0., field);

  return;
}


/** 
 * Evaluate the one-dimensional polynomial vector field at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
polynomial1D::evalField(const gsl_vector *state, gsl_vector *field)
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


/*
 * Numerical schemes definitions:
 */

/**
 * Integrate one step for a given vector field and state
 * using the scheme.
 * \param[in]     field   Vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 * \param[in]     dt      Time step.
 */
void
numericalScheme::stepForward(vectorField *field, gsl_vector *current,
			     const double dt)
{
  gsl_vector_view tmp = getStep(field, current, dt);

  // Add previous state
  gsl_vector_add(current, &tmp.vector);

  return;
}


/**
 * Get one step of integration for a given vector field and state
 * using the Euler scheme.
 * \param[in]     field   Vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 * \param[in]     dt      Time step.
 * \return                A view on the step in the workspace.
 */
gsl_vector_view
Euler::getStep(vectorField *field, gsl_vector *current, const double dt)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0); 

  // Get vector field
  field->evalField(current, &tmp.vector);
  
  // Scale by time step
  gsl_vector_scale(&tmp.vector, dt);

  return tmp;
}


/**
 * Get one step of integration for a given vector field and state
 * using the Runge-Kutta 4 scheme.
 * \param[in]     field   Vector field to evaluate.
 * \param[in,out] current Current state to update by one time step.
 * \param[in]     dt      Time step.
 * \return                A view on the step in the workspace.
 */
gsl_vector_view
RungeKutta4::getStep(vectorField *field, gsl_vector *current,
		     const double dt)
{
  /** Use views on a working matrix not to allocate memory
   *  at each time step */
  gsl_vector_view k1, k2, k3, k4, tmp; 

  // Assign views
  tmp = gsl_matrix_row(work, 0);
  k1 = gsl_matrix_row(work, 1);
  k2 = gsl_matrix_row(work, 2);
  k3 = gsl_matrix_row(work, 3);
  k4 = gsl_matrix_row(work, 4);
  
  // First increament
  field->evalField(current, &k1.vector);
  gsl_vector_scale(&k1.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k1.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, current);

  // Second increment
  field->evalField(&tmp.vector, &k2.vector);
  gsl_vector_scale(&k2.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k2.vector);
  gsl_vector_scale(&tmp.vector, 0.5);
  gsl_vector_add(&tmp.vector, current);

  // Third increment
  field->evalField(&tmp.vector, &k3.vector);
  gsl_vector_scale(&k3.vector, dt);
  
  gsl_vector_memcpy(&tmp.vector, &k3.vector);
  gsl_vector_add(&tmp.vector, current);

  // Fourth increment
  field->evalField(&tmp.vector, &k4.vector);
  gsl_vector_scale(&k4.vector, dt);

  gsl_vector_scale(&k2.vector, 2);
  gsl_vector_scale(&k3.vector, 2);
  gsl_vector_memcpy(&tmp.vector, &k1.vector);
  gsl_vector_add(&tmp.vector, &k2.vector);
  gsl_vector_add(&tmp.vector, &k3.vector);
  gsl_vector_add(&tmp.vector, &k4.vector);
  gsl_vector_scale(&tmp.vector, 1. / 6);

  return tmp;
}


/*
 * Model definitions:
 */

/**
 * Get current state.
 * param[in]  current_ Vector in which to copy the current state.
 */
void
model::getCurrentState(gsl_vector *current_)
{
  gsl_vector_memcpy(current_, current);
  
  return;
}
    
/**
 * Set current state manually.
 * \param[in] current_ Vector to copy to the current state.
 */
void
model::setCurrentState(const gsl_vector *current_)
{
  gsl_vector_memcpy(current, current_);
  
  return;
}


/**
 * Evaluate the vector field.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] vField Vector resulting from the evaluation of the vector field.
 */
void model::evalField(const gsl_vector *state, gsl_vector *vField)
{
  field->evalField(state, vField);

  return;
}


/**
 * Integrate one step the model by calling the numerical scheme.
 * \param[in]  dt Time step.
 */
void
model::stepForward(const double dt)
{
  // Apply numerical scheme to step
  scheme->stepForward(field, current, dt);
    
  return;
}


/**
 * Integrate the model for a number of time steps from the current state.
 * If data different from NULL is provided,
 * the sampled states are recorded in *data.
 * If *data is not of the right size, it is reallocated.
 * \param[in]  nt       Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  ntSpinup Initial integration steps to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrate(const size_t nt, const double dt, const size_t ntSpinup,
		 const size_t sampling, gsl_matrix **data)
{
  size_t dataSize = (size_t) ((nt - ntSpinup) / sampling + 0.1 + 1);

  // Check if data is of the right size.
  if (data && (((*data)->size1 != dataSize) || ((*data)->size2 != dim)))
    {
      gsl_matrix_free(*data);
      *data = gsl_matrix_alloc(dataSize, dim);
    }

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    stepForward(dt);

  // Save initial condition
  if (data)
    gsl_matrix_set_row(*data, 0, current);
  
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++)
    {
      stepForward(dt);

      // Save state
      if ((i%sampling == 0) && data)
	gsl_matrix_set_row(*data, (i - ntSpinup) / sampling, current);
    }

  return;
}


/**
 * Integrate the model for a number of time steps from a given inital state.
 * If data different from NULL is provided,
 * the sampled states are recorded in *data.
 * If *data is not of the right size, it is reallocated.
 * \param[in]  init     Initial state.
 * \param[in]  nt       Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  ntSpinup Initial integration steps to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrate(const gsl_vector *init, const size_t nt,
		 const double dt, const size_t ntSpinup,
		 const size_t sampling, gsl_matrix **data)
{
  // Initialize state
  setCurrentState(init);

  // Integrate
  integrate(nt, dt, ntSpinup, sampling, data);
  
  return;
}


/*
 * Integrate the model for a given period from the current state.
 * \param[in]  length   Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrate(const double length, const double dt, const double spinup,
		 const size_t sampling, gsl_matrix **data)
{
  size_t nt = (size_t) (length / dt + 0.1);
  size_t ntSpinup = (size_t) (spinup / dt + 0.1);

  integrate(nt, dt, ntSpinup, sampling, data);

  return;
}


/**
 * Integrate the model for a given period from a given initial state.
 * \param[in]  init     Initial state.
 * \param[in]  length   Duration of the integration.
 * \param[in]  dt       Time step.
 * \param[in]  spinup   Initial integration period to remove.
 * \param[in]  sampling Time step at which to save states.
 * \param[out] data     Record states in pointed data.
 */
void
model::integrate(const gsl_vector *init, const double length, const double dt,
		 const double spinup, const size_t sampling, gsl_matrix **data)
{
  size_t nt = (size_t) (length / dt + 0.1);
  size_t ntSpinup = (size_t) (spinup / dt + 0.1);

  integrate(init, nt, dt, ntSpinup, sampling, data);

  return;
}

/*
 * Linearized Model definitions:
 */

/**
 * Get current state.
 * param[in]  current_ Matrix in which to copy the current state.
 */
void fundamentalMatrixModel::getCurrentState(gsl_matrix *current_)
{
  gsl_matrix_memcpy(current_, current);
  
  return;
}
    
/**
 * Integrate one step the linearized model by calling the numerical scheme.
 * First the full model is integrated.
 * Then, integrate the linearized model.
 * Last, the Jacobian matrix is updated to the new state.
 * \param[in] dt Time step.
 */
void
fundamentalMatrixModel::stepForward(const double dt)
{
  gsl_vector_view col;

  // Integrate the model
  mod->stepForward(dt);

  // Integrate the fundamental matrix
  for (size_t d = 0; d < dim; d++)
    {
      // Get matrix column
      col = gsl_matrix_column(current, d); 
  
      // Apply numerical scheme to step
      scheme->stepForward(Jacobian, &col.vector, dt);
    }
    
  // Update the Jacobian with the full model current state
  Jacobian->setMatrix(mod->current);
  
  return;
}


/**
 * Integrate the fundamentalMatrixModel for a given number
 * of time steps and from the current state.
 * \param[in]  nt      Number of time steps to integrate.
 * \param[in]  dt      Time step.
 */
void
fundamentalMatrixModel::integrate(const size_t nt, const double dt)
{
  // Get record
  for (size_t i = 1; i <= nt; i++)
      stepForward(dt);

  return;
}


/**
 * Integrate the fundamentalMatrixModel for a number of time steps
 * and from a given initial state.
 * \param[in]  init    Initial state.
 * \param[in]  initMat Initial fundamental matrix.
 * \param[in]  nt      Number of time steps to integrate.
 * \param[in]  dt      Time step.
 */
void
fundamentalMatrixModel::integrate(const gsl_vector *init,
					 const gsl_matrix *initMat,
					 const size_t nt, const double dt)
{
  // Initialize state and fundamental matrix
  setCurrentState(init, initMat);

  // Integrate
  integrate(nt, dt);

  return;
}


/**
 * Integrate the fundamentalMatrixModel for a given period
 * and from the current state.
 * \param[in]  length      Duration of the integration.
 * \param[in]  dt          Time step.
 */
void
fundamentalMatrixModel::integrate(const double length, const double dt)
{
  size_t nt = (size_t) (length / dt + 0.1);

  // Get record
  integrate(nt, dt);

  return;
}

/**
 * Integrate the fundamentalMatrixModel for a given period
 * and from a given initial state.
 * \param[in]  init        Initial state.
 * \param[in]  initMatrix  Initial fundamental matrix.
 * \param[in]  length      Duration of the integration.
 * \param[in]  dt          Time step.
 */
void
fundamentalMatrixModel::integrate(const gsl_vector *init,
				  const gsl_matrix *initMat,
				  const double length, const double dt)
{
  size_t nt = (size_t) (length / dt + 0.1);

  // Get record
  integrate(init, initMat, nt, dt);

  return;
}

/**
 * Get the fundamental matrices Mts for s between 0 and t.
 * \param[in]  nt       Number of time steps to integrate.
 * \param[in]  dt       Time step.
 * \param[out] xt       Matrix in which to record states.
 * \param[out] Mts      Vector of matrices in which to record state matrices.
 * \param[in]  ntSpinup Initial integration steps to remove.
 */
void
fundamentalMatrixModel::integrateRange(const size_t nt, const double dt,
				       gsl_matrix **xt,
				       std::vector<gsl_matrix *> *Mts,
				       const size_t ntSpinup)
{
  size_t nSamp = (size_t) ((nt - ntSpinup) + 0.1 + 1);
  gsl_matrix *stepMat = gsl_matrix_alloc(dim, dim);
  gsl_vector_view colInit, colFinal;

  // Resize Mts vector to the right size
  Mts->resize(nSamp);
  
  // Check if xt is of the right size.
  if (xt && (((*xt)->size1 != nSamp) || ((*xt)->size2 != dim))) {
    gsl_matrix_free(*xt);
    *xt = gsl_matrix_alloc(nSamp, dim);
  }

  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++)
    mod->stepForward(dt);
  // Update the Jacobian after spinup and set current to identity
  setCurrentState();
  
  // Save initial conditions
  gsl_matrix_set_row(*xt, 0, mod->current);
  Mts->at(0) = gsl_matrix_alloc(dim, dim);
  gsl_matrix_set_identity(Mts->at(0)); // Should be the same as current
  
  // Get record
  for (size_t r = 1; r < nSamp; r++) {
    // Initialize fundamental matrix Mtr
    Mts->at(r) = gsl_matrix_alloc(dim, dim);
    gsl_matrix_set_identity(Mts->at(r));
    
    // Get matrix of stepMat r * dt
    for (size_t d = 0; d < dim; d++) {
      // Get matrix column
      colInit = gsl_matrix_column(current, d); 

      // Apply numerical scheme to current matrix
      colFinal = scheme->getStep(Jacobian, &colInit.vector, dt);

      // Set stepMat matrix
      colInit = gsl_matrix_column(stepMat, d);
      gsl_vector_memcpy(&colInit.vector, &colFinal.vector);
    }

    // Add step matrix to Mttau with tau < r
    for (size_t tau = 0; tau < r; tau++)
      gsl_matrix_add(Mts->at(tau), stepMat);

    // Step the current state forward
    mod->stepForward(dt);
    
    // Step to current fundamental matrix forward
    gsl_matrix_add(current, stepMat);

    // Update the Jacobian with the full model current state
    Jacobian->setMatrix(mod->current);

    // Save state
    gsl_matrix_set_row(*xt, r, mod->current);
  }

  // Free
  gsl_matrix_free(stepMat);

  return;
}


/**
 * Get the fundamental matrices Mts for s between 0 and t for a given
 * initial state.
 * \param[in]  init     Initial state.
 * \param[in]  nt       Number of time steps to integrate.
 * \param[in]  dt       Time step.
 * \param[out] xt       Matrix in which to record states.
 * \param[out] Mts      Vector of matrices in which to record state matrices.
 * \param[in]  ntSpinup Initial integration steps to remove.
 */
void
fundamentalMatrixModel::integrateRange(const gsl_vector *init,
				       const size_t nt, const double dt,
				       gsl_matrix **xt,
				       std::vector<gsl_matrix *> *Mts,
				       const size_t ntSpinup)
{
  // Initialize state and fundamental matrix to identity
  setCurrentState(init);

  // Integrate
  integrateRange(nt, dt, xt, Mts, ntSpinup);

  return;
}


/**
 * Set current fundamental matrix manually.
 * \param[in] currentMat Fundamental matrix to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_matrix *currentMat)
{
  gsl_matrix_memcpy(current, currentMat);
  
  return;
}
    
/**
 * Set current state and fundamental matrix manually.
 * \param[in]  current_   Current state to set to.
 * \param[in]  currentMat Current fundamental matrix to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_vector *current_,
					const gsl_matrix *currentMat)
{
  // Set current state of model
  mod->setCurrentState(current_);
  
  // Update Jacobian to that at current state
  Jacobian->setMatrix(mod->current);
  
  // Update current state of the fundamental matrix
  setCurrentState(currentMat);
  
  return;
}
    

/**
 * Update Jacobian to current model state and set fundamental matrix to identity.
 */
void
fundamentalMatrixModel::setCurrentState()
{
  // Update Jacobian to that at current state
  Jacobian->setMatrix(mod->current);
  
  // Set fundamental matrix to identity
  gsl_matrix_set_identity(current);
  
  return;
}
    

/**
 * Set current state manually and fundamental matrix to identity.
 * \param[in]  current_ Current state to set to.
 */
void
fundamentalMatrixModel::setCurrentState(const gsl_vector *current_)
{
  // Set current state of model
  mod->setCurrentState(current_);

  // Update Jacobian to current model state
  // and set fundamental matrix to identity
  setCurrentState();
  
  return;
}
    
