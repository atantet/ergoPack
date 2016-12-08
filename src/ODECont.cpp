#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl_extension.hpp>
#include <math.h>
#include <iostream>


/** \file ODECont.cpp
 *  \brief Definitions for ODECont.hpp
 *
 */

/*
 * Method definitions:
 */

/*
 * Abstract solutionCorrection definitions:
 */

/**
 * Get current state of tracking.
 * \param[in]  current_ Vector in which to copy the current state.
 */
void
solutionCorrection::getCurrentState(gsl_vector *current_)
{
  gsl_vector_memcpy(current_, current);

  return;
}

/** 
 * Update state after Newton step.
 * \param[in] damping Damping to apply to the Newton step.
 */
void
solutionCorrection::applyCorr(const double damping)
{
  // Damp stepCorr
  gsl_vector_scale(stepCorr, damping);

  // Update current state
  gsl_vector_add(current, stepCorr);
  
  return;
}


/**
 * Perform a Newton step to find a fixed point.
 */
void
solutionCorrection::NewtonStep()
{
  gsl_permutation *p = gsl_permutation_alloc(S->size2);
  int s;

  // Get matrix of the linear system
  getSystemCorr();

  // Set up LU decomposition
  // Note that this modifies the matrix S
  gsl_linalg_LU_decomp(S, p, &s);

  // Solve the linear system
  gsl_linalg_LU_solve(S, p, targetCorr, stepCorr);

  return;
}


/*
 * Abstract fixedPointTrack definitions:
 */

/**
 * Set current state of the model and Jacobian from the current state.
 */
void
fixedPointTrack::setCurrentState()
{
  // Evaluate Jacobian at current state.
  Jacobian->setMatrix(current);

  return;
}


/**
 * Set current state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
fixedPointTrack::setCurrentState(const gsl_vector *init)
{
  // Set current state
  gsl_vector_memcpy(current, init);

  // Set state of model
  setCurrentState();

  return;
}


/**
 * Get Jacobian matrix of the current state.
 * \param[in]  Matrix in which to copy the Jacobian.
 */
void
fixedPointTrack::getStabilityMatrix(gsl_matrix *matrix)
{
  // Set matrix at current state
  setCurrentState();
  
  // Save the matrix
  Jacobian->getMatrix(matrix);
  
  return;
}


/*
 * Fixed point correction definitions:
 */

/**
 * Get matrix of the linear system to be solved.
 */
void
fixedPointCorr::getSystemCorr()
{
  // Store the Jacobian in S
  Jacobian->getMatrix(S);
		      
  return;
}

/**
 * Update targetCorr vector for fixed point tracking.
 */
void
fixedPointCorr::updateTargetCorr()
{
  // Update targetCorr
  field->evalField(current, targetCorr);
  gsl_vector_scale(targetCorr, -1.);

  return;
}


/**
 * Find fixed point using Newton-Raphson method.
 * \param[in]  init Vector from which to start tracking.
 */
void
fixedPointCorr::findSolution(const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);  
    
  // Get targetCorr vector -f(x)
  updateTargetCorr();
  
  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      // Perform Newton step
      NewtonStep();

      // Update stepCorr size before to damp it
      errStepCorrSize = gsl_vector_get_norm(stepCorr);

      // Update model state
      applyCorr();

      // Get targetCorr vector -f(x)
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state */
      setCurrentState();

      // Update distance to targetCorr and iterate
      errDist = gsl_vector_get_norm(targetCorr);
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

  return;
}


/*
 * Fixed point continuation definitions:
 */

/**
 * Get matrix of the linear system to be solved for correction.
 */
void
fixedPointCont::getSystemCorr()
{
  gsl_vector_view vView;

  // Set the matrix to Jacobian, the last row will be modified.
  Jacobian->getMatrix(S);
  
  // Set the lower part to the previous prediction step
  vView = gsl_matrix_subrow(S, dim-1, 0, dim);
  gsl_vector_memcpy(&vView.vector, stepPred);

  return;
}


/**
 * Get matrix of the linear system to be solved for prediction.
 */
void
fixedPointCont::getSystemPred()
{
  getSystemCorr();

  return;
}


/**
 * Prediction Newton-Raphson step.
 */
void
fixedPointCont::predict()
{
  gsl_permutation *p = gsl_permutation_alloc(dim);
  int s;

  // Get linear system
  getSystemPred();

  // Set up LU decomposition
  // Note that this modifies the matrix A
  gsl_linalg_LU_decomp(S, p, &s);

  // Solve the linear system to get new predition step
  gsl_linalg_LU_solve(S, p, targetPred, stepPred);

  return;
}


/** 
 * Update state after prediction
 */
void
fixedPointCont::applyPredict(const double contStep)
{
  // Scale prediction step by continuation step
  gsl_vector_scale(stepPred, contStep);
  
  // Update current state
  gsl_vector_add(current, stepPred);

  // Scale back, since the continuation step is used as later
  gsl_vector_scale(stepPred, 1. / contStep);

  return;
}


/**
 * Update correction target vector for fixed point tracking.
 */
void
fixedPointCont::updateTargetCorr()
{
  // Set state target, last element will be modified
  field->evalField(current, targetCorr);
  gsl_vector_scale(targetCorr, -1.);

  // Set target for pseudo-arclength
  gsl_vector_set(targetCorr, dim-1, 0.);

  return;
}


/**
 * Correct prediction by pseudo-arclenght continuation and Newton-Raphson.
 */
void
fixedPointCont::correct()
{
  // Make sure the iteration counter and errors are reset.
  numIter = 0;
  errDist = 1.e27;
  errStepCorrSize = 1.e27;
  converged = false;
  
  // Get target vector -f(x)
  updateTargetCorr();

  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      // Perform Newton step
      NewtonStep();

      // Update step size before to damp it
      errStepCorrSize = gsl_vector_get_norm(stepCorr);

      // Update model state
      applyCorr();

      // Get target vector -f(x)
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state */
      setCurrentState();

      // Update distance to target and iterate
      errDist = gsl_vector_get_norm(targetCorr);
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

  return;
}


/**
 * Correct initial state.
 */
void
fixedPointCont::correct(const gsl_vector *init)
{
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);

  // Correct
  correct();

}


/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from current step.
 * \param[in]  contStep Continuation step size.
 */
void
fixedPointCont::continueStep(const double contStep)
{
  numIter = 0;
  
  // Initialize to current state
  setCurrentState();

  // Predict
  predict();

  // Apply prediction
  applyPredict(contStep);

  // Update Jacobian matrix (the prediction has been applied)
  setCurrentState();
  
  // Correct using Newton-Raphson
  correct();
  
  return;
}

/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from initial state.
 * \param[in]  contStep Continuation step size.
 * \param[in]  init     Vector from which to start tracking.
 */
void
fixedPointCont::continueStep(const double contStep, const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);

  // Continue
  continueStep(contStep);
  
  return;
}


/*
 * Abstract periodic orbit tracking definitions:
 */

/**
 * Get current state of tracking.
 * If current_->size == dim + 1, return x(0), T
 * else if current_->size == dim*numShoot + 1, return x(0), ..., x(numShoot-1), T
 * \param[in]  current_ Vector in which to copy the current state.
 */
void
periodicOrbitTrack::getCurrentState(gsl_vector *current_)
{
  if (current_->size == dim + 1)
    {
      // Set state x(0) (dim element will be changed)
      gsl_vector_view vView = gsl_vector_subvector(current, 0, dim+1);
      gsl_vector_memcpy(current_, &vView.vector);
      // Set the period
      gsl_vector_set(current_, dim, gsl_vector_get(current, dim * numShoot));
    }
  else if (current_->size == dim * numShoot + 1)
    gsl_vector_memcpy(current_, current);
  else
    std::cerr << "Destination vector size " << current_->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1) << std::endl;

  return;
}

/**
 * Adapt time step and number of time steps to shooting strategy.
 */
void
periodicOrbitTrack::adaptTimeToPeriod()
{
  double T;
  
  // Get integration time stepCorr.
  T = gsl_vector_get(current, dim * numShoot);

  // Adapt
  adaptTimeToPeriod(T);
  
  return;
}


/**
 * Adapt time step and number of time steps to shooting strategy.
 * \param[in]  T Period to adapt time to.
 */
void
periodicOrbitTrack::adaptTimeToPeriod(const double T)
{
  // Get total number of time steps
  nt = (size_t) (ceil(T / intStepCorr) + 0.1);
  
  // Get time step
  dt = T / nt;
  
  // Uniformly divide number of time steps
  gsl_vector_uint_set_all(ntShoot, (size_t) (nt / numShoot));
  
  // Add remaining
  gsl_vector_uint_set(ntShoot, numShoot-1,
		      gsl_vector_uint_get(ntShoot, numShoot-1) + (nt % numShoot));

  return;
}


/**
 * Set current state of the model and fundamental matrix
 */
void
periodicOrbitTrack::setCurrentState()
{
  // Set state of linearized model, i.e. the first dim entries of current,
  // even when multiple shooting.
  gsl_vector_const_view currentState = gsl_vector_const_subvector(current, 0, dim);
  linMod->setCurrentState(&currentState.vector);

  return;
}


/**
 * Set current state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
periodicOrbitTrack::setCurrentState(const gsl_vector *init)
{
  gsl_vector_view currentState;
  
  // Set current state
  // Check if initCont as size dim+1 or dim*numShoot+1
  if (init->size == dim + 1)
    {
      // Set period
      gsl_vector_set(current, dim*numShoot, gsl_vector_get(init, dim));

      // Adapt time
      adaptTimeToPeriod();
      
      // Set x(0) and initialize model
      currentState = gsl_vector_subvector(current, 0, dim);
      gsl_vector_const_view vView = gsl_vector_const_subvector(init, 0, dim);
      gsl_vector_memcpy(&currentState.vector, &vView.vector);
      linMod->setCurrentState(&vView.vector);

      // Only x(0) has been given, integrate to x(s) and set
      for (size_t s = 0; s < numShoot - 1; s++)
	{
	  linMod->mod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);
	  currentState = gsl_vector_subvector(current, (s+1)*dim, dim);
	  linMod->mod->getCurrentState(&currentState.vector);
	}
    }
  else if (init->size == dim * numShoot + 1)
    gsl_vector_memcpy(current, init);
  else
    std::cerr << "Initial state size " << init->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1) << std::endl;

  // Set current state of model
  setCurrentState();

  return;
}


/**
 * Get fundamental matrix of the current state.
 * \param[in]  Matrix in which to copy the fundamental matrix.
 */
void
periodicOrbitTrack::getStabilityMatrix(gsl_matrix *matrix)
{
  // Update model current state
  setCurrentState();
  
  // Get integration time stepCorr.
  adaptTimeToPeriod();

  // Get solution and fundamental matrix after the period
  linMod->integrateForward(nt, dt);

  // Copy
  linMod->getCurrentState(matrix);
  
  // Rewind
  setCurrentState();
  
  return;
}


/*
 * Periodic orbit correction definitions:
 */

/**
 * Get matrix of the linear system to be solved.
 */
void
periodicOrbitCorr::getSystemCorr()
{
  gsl_matrix_view mView;
  gsl_vector_view vView, vView2;

  // Set initial data
  gsl_matrix_set_zero(S);
  for (size_t s = 0; s < numShoot; s++)
    {
      // Add F(x(s)) to last row
      // Get view of the last row
      vView = gsl_matrix_subrow(S, dim*numShoot, s*dim, dim);
      // Get view of the state x(s) and set linMod to it
      // (used to integrate to (x(s+T), M(s+T)))
      vView2 = gsl_vector_subvector(current, s*dim, dim);
      linMod->setCurrentState(&vView2.vector);
      // Evaluate F(x(s)) to the last row
      linMod->mod->evalField(linMod->mod->current, &vView.vector);
  
      // Set identity matrices
      mView = gsl_matrix_submatrix(S, s*dim, ((s+1) % numShoot)*dim, dim, dim);
      gsl_matrix_set_identity(&mView.matrix);
      gsl_matrix_scale(&mView.matrix, -1.);
  
      // Get solution x(s + T) and fundamental matrix M(s + T)
      linMod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);

      // Add M(s + t) to state part
      mView = gsl_matrix_submatrix(S, s*dim, s*dim, dim, dim);
      // Used add in case numShoot == 1
      gsl_matrix_add(&mView.matrix, linMod->current);
    
      // Add F(x(s + t)) to last column
      vView = gsl_matrix_subcolumn(S, dim*numShoot, s*dim, dim);
      linMod->mod->evalField(linMod->mod->current, &vView.vector);
    }

  return;
}


/**
 * Update targetCorr vector for periodic orbit tracking.
 */
void
periodicOrbitCorr::updateTargetCorr()
{
  gsl_vector_view targetCorrState, currentState;

  for (size_t s = 0; s < numShoot; s++)
    {
      // Get a view on x(s+1)
      currentState = gsl_vector_subvector(current, ((s+1) % numShoot)*dim, dim);
      // Get a view on part of target (s)
      targetCorrState = gsl_vector_subvector(targetCorr, s*dim, dim);
      // Copy the current state x(s+1) there
      gsl_vector_memcpy(&targetCorrState.vector, &currentState.vector);
  
      /** Integrate model for a period from the new current state
	  after initializing the linear model to x(s)
       *  (no need to update the linear model here). */
      currentState = gsl_vector_subvector(current, s*dim, dim);
      linMod->setCurrentState(&currentState.vector);
      linMod->mod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);

      // Substract the new state
      gsl_vector_sub(&targetCorrState.vector, linMod->mod->current);
    }
  
  // Set targetCorr for period to 0
  gsl_vector_set(targetCorr, dim * numShoot, 0.);

  return;
}


/**
 * Find periodic orbit using Newton-Raphson method.
 * \param[in]  init Vector from which to start tracking.
 */
void
periodicOrbitCorr::findSolution(const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize the current state of tracking and that of the linearized model.
  setCurrentState(init);

  // Set time step
  adaptTimeToPeriod();

  // Get targetCorr vector to x - \phi_T(x) and update error
  updateTargetCorr();
  errDist = gsl_vector_get_norm(targetCorr) / numShoot;

  /** Update state x of model and Jacobian J(x) to current state
   *  and reinitizlize fundamental matrix to identity
   *  (after integration in updateTargetCorr). */
  setCurrentState();
    
  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      // Perform Newton step (leaves the model integrated forward)
      NewtonStep();

      // Update correction step size
      errStepCorrSize = gsl_vector_get_norm(stepCorr) / numShoot;

      // Update model state
      applyCorr();

      // Set time step
      adaptTimeToPeriod();

      // Get targetCorr vector to x - \phi_T(x) and state of the linearized model
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      setCurrentState();
    
      // Update distance to targetCorr
      errDist = gsl_vector_get_norm(targetCorr) / numShoot;
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;
  
  return;
  
}


/*
 * Periodic orbit continuation definitions
 */

/**
 * Adapt time step and number of time steps to shooting strategy.
 * \param[in]  T Period to adapt time to.
 */
void
periodicOrbitCont::adaptTimeToPeriod(const double T)
{
  // Get total number of time steps
  nt = (size_t) (ceil(T / intStepCorr) + 0.1);
  
  // Get time step
  dt = T / nt;
  
  // Uniformly divide number of time steps
  gsl_vector_uint_set_all(ntShoot, (size_t) (nt / numShoot));
  
  // Add remaining
  gsl_vector_uint_set(ntShoot, numShoot-1,
		      gsl_vector_uint_get(ntShoot, numShoot-1) + (nt % numShoot));

  return;
}


/**
 * Adapt time step and number of time steps to shooting strategy.
 */
void
periodicOrbitCont::adaptTimeToPeriod()
{
  double T;
  
  // Get integration time stepCorr.
  T = gsl_vector_get(current, (dim-1) * numShoot + 1);

  // Adapt
  adaptTimeToPeriod(T);
  
  return;
}


/**
 * Get extended state vector x(s), lambda.
 * \param[out] state Vector in which to copy the extended state.
 * \param[in]  s     Shooting number. 
 */
void
periodicOrbitCont::getExtendedState(gsl_vector *state, const size_t s)
{
  gsl_vector_view vView1, vView2;
  
  vView1 = gsl_vector_subvector(current, s*(dim-1), dim-1);
  vView2 = gsl_vector_subvector(state, 0, dim-1);
  gsl_vector_memcpy(&vView2.vector, &vView1.vector);
  gsl_vector_set(state, dim-1, gsl_vector_get(current, (dim-1) * numShoot));
  
  return;
}


/**
 * Get current state of tracking.
 * If current_->size == dim + 1, return x(0), T
 * else if current_->size == dim*numShoot + 1, return x(0), ..., x(numShoot-1), T
 * \param[in]  current_ Vector in which to copy the current state.
 */
void
periodicOrbitCont::getCurrentState(gsl_vector *current_)
{
  if (current_->size == dim + 1)
    {
      // Set state x(0)
      gsl_vector_view vView = gsl_vector_subvector(current, 0, dim);
      gsl_vector_view vView2 = gsl_vector_subvector(current_, 0, dim);
      gsl_vector_memcpy(&vView2.vector, &vView.vector);

      // Set parameter
      gsl_vector_set(current_, dim-1, gsl_vector_get(current, (dim-1)*numShoot));
      
      // Set the period
      gsl_vector_set(current_, dim, gsl_vector_get(current, (dim-1) * numShoot + 1));
    }
  else if (current_->size == dim * numShoot + 1)
    gsl_vector_memcpy(current_, current);
  else
    std::cerr << "Destination vector size " << current_->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1) << std::endl;

  return;
}

/**
 * Set current state x(s), lambda of the model and fundamental matrix
 * \param[in]  s Shooting number
 */
void
periodicOrbitCont::setCurrentState(const size_t s)
{
  // Set state of linearized model x(0), lambda.
  getExtendedState(work, s);
  linMod->setCurrentState(work);

  return;
}


/**
 * Set current state x(0), lambda of the model and fundamental matrix
 */
void
periodicOrbitCont::setCurrentState()
{
  // Set state of linearized model x(0), lambda.
  setCurrentState((size_t) 0);

  return;
}


/**
 * Set current state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
periodicOrbitCont::setCurrentState(const gsl_vector *init)
{
  gsl_vector_view currentState;
  
  // Set current state
  // Check if initCont as size dim+1 or dim*numShoot+1
  if (init->size == dim + 1)
    {
      // Set period (to adapt time)
      gsl_vector_set(current, (dim-1)*numShoot+1, gsl_vector_get(init, dim));

      // Adapt time
      adaptTimeToPeriod();
      
      // Set x(0), lambda and initialize model (last element will be overwritten)
      currentState = gsl_vector_subvector(current, 0, dim);
      gsl_vector_const_view vView = gsl_vector_const_subvector(init, 0, dim);
      gsl_vector_memcpy(&currentState.vector, &vView.vector);
      linMod->setCurrentState(&vView.vector);

      // Only x(0) has been given, integrate to x(s) and set
      for (size_t s = 0; s < numShoot - 1; s++)
	{
	  linMod->mod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);
	  currentState = gsl_vector_subvector(current, (s+1)*(dim-1), dim);
	  linMod->mod->getCurrentState(&currentState.vector);
	}
      
      // Set period (again, because overwritten)
      gsl_vector_set(current, (dim-1)*numShoot+1, gsl_vector_get(init, dim));
    }
  else if (init->size == (dim-1) * numShoot + 2)
    gsl_vector_memcpy(current, init);
  else
    std::cerr << "Initial state size " << init->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1) << std::endl;
  // Set current state of model
  setCurrentState();

  return;
}


/**
 * Get matrix of the linear system to be solved for correction.
 */
void
periodicOrbitCont::getSystemCorr()
{
  gsl_matrix_view mView, mView2;
  gsl_vector_view vView, vView2;

  // Set initial data
  gsl_matrix_set_zero(S);
  for (size_t s = 0; s < numShoot; s++)
    {
      // Add F(x(s)) to last row (the last entry will be updated, with the very last 0)
      // S[dim*numShoot+1, :dim*numShoot+1] = F(x(0)), ... F(x(numShoot-1))
      // with F[dim * numShoot] = 0.
      vView = gsl_matrix_subrow(S, (dim-1)*numShoot+1, s*(dim-1), dim);
      // Get view of the state x(s) and set linMod to it
      // (used to integrate to (x(s+T), M(s+T)))
      setCurrentState(s);
      linMod->mod->evalField(linMod->mod->current, &vView.vector);

      // Set identity matrices
      mView = gsl_matrix_submatrix(S, s*(dim-1), ((s+1) % numShoot)*(dim-1),
				   dim-1, dim-1);
      gsl_matrix_set_identity(&mView.matrix);
      gsl_matrix_scale(&mView.matrix, -1.);

      // Get solution x(s + T) and fundamental matrix M(s + T)
      linMod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);

      // Set data after a period
      // Add M(s + T) to state part
      mView = gsl_matrix_submatrix(S, s*(dim-1), s*(dim-1), dim-1, dim-1);
      mView2 = gsl_matrix_submatrix(linMod->current, 0, 0, dim-1, dim-1);
      // Used add in case numShoot == 1
      gsl_matrix_add(&mView.matrix, &mView2.matrix);

      // Set the column for the derivative with respect to parameter
      vView = gsl_matrix_subcolumn(S, numShoot*(dim-1), s*(dim-1), dim-1);
      vView2 = gsl_matrix_subcolumn(linMod->current, dim-1, 0, dim-1);
      gsl_vector_memcpy(&vView.vector, &vView2.vector);
      
      // Set F(x(s + T)) last column 
      vView = gsl_matrix_subcolumn(S, (dim-1)*numShoot+1, s*(dim-1), dim);
      linMod->mod->evalField(linMod->mod->current, &vView.vector);
    }
  
  // Set the penultiem row to previous prediction step to normalize the step.
  vView = gsl_matrix_row(S, (dim-1)*numShoot);
  gsl_vector_memcpy(&vView.vector, stepPred);
  // Set last element to zero to exclude period from normalization
  gsl_matrix_set(S, (dim-1)*numShoot, (dim-1)*numShoot+1, 0.);

  // Set bottom right corner to zero
  gsl_matrix_set(S, (dim-1)*numShoot+1, (dim-1)*numShoot+1, 0.);

  return;
}


/**
 * Update correction target vector for fixed point tracking.
 */
void
periodicOrbitCont::updateTargetCorr()
{
  gsl_vector_view targetCorrState;

  for (size_t s = 0; s < numShoot; s++)
    {
      // Get a view on part of target (s) (the last element will be overwritten)
      targetCorrState = gsl_vector_subvector(targetCorr, s*(dim-1), dim);
      
      // Get extended state vector x(s+1), lambda and copy it to target (s)
      getExtendedState(work, ((s + 1) % numShoot));
      gsl_vector_memcpy(&targetCorrState.vector, work);
  
      /** Integrate model for a period from the new current state
	  after initializing the linear model to x(s)
       *  (no need to update the linear model here). */
      setCurrentState(s);
      linMod->mod->integrateForward((size_t) gsl_vector_uint_get(ntShoot, s), dt);

      // Substract the new state
      gsl_vector_sub(&targetCorrState.vector, linMod->mod->current);
    }
  
  // Set targetCorr for parameter to 0 (pseudo-arclength)
  gsl_vector_set(targetCorr, (dim-1) * numShoot, 0.);

  // Set targetCorr for period to 0
  gsl_vector_set(targetCorr, (dim-1) * numShoot + 1, 0.);

  return;
}


/**
 * Get matrix of the linear system to be solved for prediction.
 */
void
periodicOrbitCont::getSystemPred()
{
  getSystemCorr();

  return;
}


/**
 * Prediction Newton-Raphson step.
 */
void
periodicOrbitCont::predict()
{
  gsl_permutation *p = gsl_permutation_alloc((dim-1)*numShoot + 2);
  int s;

  // Get linear system
  getSystemPred();

  // Set up LU decomposition
  // Note that this modifies the matrix A
  gsl_linalg_LU_decomp(S, p, &s);

  // Solve the linear system to get new predition step
  gsl_linalg_LU_solve(S, p, targetPred, stepPred);

  return;
}


/** 
 * Update state after prediction
 */
void
periodicOrbitCont::applyPredict(const double contStep)
{
  // Scale prediction step by continuation step
  gsl_vector_scale(stepPred, contStep);
  
  // Update current state
  if (verbose)
    {
      std::cout << "Applying prediction:" << std::endl;
      gsl_vector_fprintf(stdout, stepPred, "%lf");
    }
  gsl_vector_add(current, stepPred);

  // Scale back, since the continuation step is used as later
  gsl_vector_scale(stepPred, 1. / contStep);

  return;
}


/**
 * Correct prediction by pseudo-arclenght continuation and Newton-Raphson.
 */
void
periodicOrbitCont::correct()
{
  // Make sure the iteration counter and errors are reset.
  numIter = 0;
  errDist = 1.e27;
  errStepCorrSize = 1.e27;
  converged = false;
  
  // Set time step
  adaptTimeToPeriod();

  // Get targetCorr vector to x - \phi_T(x) and update error
  errDist = gsl_vector_get_norm(targetCorr) / numShoot;
  updateTargetCorr();

  /** Update state x of model and Jacobian J(x) to current state
   *  and reinitizlize fundamental matrix to identity
   *  (after integration in updateTargetCorr). */
  setCurrentState();
    
  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      if (verbose)
	{
	  std::cout << "Correction iteration " << numIter << " with previous step:\n"
		    << "Current state: " << std::endl;
	  gsl_vector_fprintf(stdout, current, "%lf");
	  std::cout << "Correction step: " << std::endl;
	  gsl_vector_fprintf(stdout, stepCorr, "%lf");
	  std::cout << "errDist: " << errDist << "\nerrStepCorrSize = "
		    << errStepCorrSize << std::endl;
	}
      
      // Perform Newton step
      NewtonStep();
      
      // Update step size before to damp it
      errStepCorrSize = gsl_vector_get_norm(stepCorr) / numShoot;

      // Update model state
      applyCorr();

      // Set time step
      adaptTimeToPeriod();

      // Get target vector
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      setCurrentState();
    
      // Update distance to target and iterate
      errDist = gsl_vector_get_norm(targetCorr) / numShoot;
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

  return;
}


/**
 * Correct initial state.
 */
void
periodicOrbitCont::correct(const gsl_vector *init)
{
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);

  // Correct
  correct();

  return;
}


/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from current step.
 * \param[in]  contStep Continuation step size.
 */
void
periodicOrbitCont::continueStep(const double contStep)
{
  numIter = 0;
  
  // Initialize to current state
  setCurrentState();

  // Previous prediction
  if (verbose)
    {
      std::cout << "Previous prediction step:" << std::endl;
      gsl_vector_fprintf(stdout, stepPred, "%lf");
    }

  // Predict
  predict();

  // Apply prediction
  applyPredict(contStep);

  // Update model and Jacobian to current state (the prediction has been applied)
  setCurrentState();
  
  // Correct using Newton-Raphson
  if (verbose)
    std::cout << "Correcting..." << std::endl;
  correct();
  
  return;
}

/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from initial state.
 * \param[in]  contStep Continuation step size.
 * \param[in]  init     Vector from which to start tracking.
 */
void
periodicOrbitCont::continueStep(const double contStep, const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);

  // Continue
  continueStep(contStep);
  
  return;
}


