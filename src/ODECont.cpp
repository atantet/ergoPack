#include <iostream>
#include <cmath>
#include <cstring>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_eigen.h>
#include <gsl_extension.hpp>


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
 * Get initial state of tracking.
 * \param[in]  initial_ Vector in which to copy the initial state.
 */
void
solutionCorrection::getInitialState(gsl_vector *initial_)
{
  gsl_vector_memcpy(initial_, initial);

  return;
}

/** 
 * Update state after Newton step.
 * \param[in] damping Damping to apply to the Newton step.
 */
void
solutionCorrection::applyCorr(const double damping)
{
  if (verbose)
    std::cout << "Applying correction..." << std::endl;
  
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

  if (verbose)
    std::cout << "Newton step..." << std::endl;
  
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
  if (verbose)
    std::cout << "Setting current state..." << std::endl;
  
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
 * Set initial state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
fixedPointTrack::setInitialState(const gsl_vector *init)
{
  // Set initial state
  gsl_vector_memcpy(initial, init);

  // Set current state
  initialize();

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
  if (verbose)
    std::cout << "Updating target..." << std::endl;
  
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
  if (verbose)
    std::cout << "Looking for solution..." << std::endl;
  
  numIter = 0;
  
  // Initialize tracking and linearized model state
  setInitialState(init);  
    
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
 * General continuation definitions:
 */

/**
 * Correct initial state.
 */
void
solutionCont::correct(const gsl_vector *init)
{
  // Initialize current state of tracking and linearized model state
  setInitialState(init);

  // Correct
  correct();

  return;
}


/** 
 * Update state after prediction
 */
void
solutionCont::applyPredict(const double contStep, gsl_vector *current)
{
  // Scale prediction step by continuation step
  gsl_vector_scale(stepPred, contStep);
  
  // Update current state
  if (verboseCont) {
      std::cout << "Applying prediction:" << std::endl;
      gsl_vector_fprintf(stdout, stepPred, "%lf");
  }
  gsl_vector_add(current, stepPred);

  // Scale back, since the continuation step is used as later
  gsl_vector_scale(stepPred, 1. / contStep);

  return;
}


/**
 * Prediction Newton-Raphson step.
 */
void
solutionCont::predict(gsl_matrix *S)
{
  gsl_permutation *p;
  int s;

  if (verboseCont)
    std::cout << "Predicting..." << std::endl;
  
  // Get linear system
  getSystemPred();

  // Set up LU decomposition
  // Note that this modifies the matrix A
  p = gsl_permutation_alloc(S->size1);
  gsl_linalg_LU_decomp(S, p, &s);

  // Solve the linear system to get new predition step
  gsl_linalg_LU_solve(S, p, targetPred, stepPred);

  return;
}


/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from initial state.
 * \param[in]  contStep Continuation step size.
 * \param[in]  init     Vector from which to start tracking.
 */
void
solutionCont::continueStep(const double contStep, const gsl_vector *init)
{
  // Initialize current state of tracking and linearized model state
  setInitialState(init);

  // Continue
  continueStep(contStep);
  
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
 * Update correction target vector for fixed point tracking.
 */
void
fixedPointCont::updateTargetCorr()
{
  if (verbose)
    std::cout << "Updating target..." << std::endl;
  
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
  if (verbose)
    std::cout << "Correcting..." << std::endl;
  
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
      if (verbose)
	{
	  std::cout << "Correction iteration " << numIter
		    << " with preveious step:\n"
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

      if (verbose)
	std::cout << "" << std::endl;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

  return;
}


/**
 * Perform one step (correction + prediction) of pseudo-arclength continuation
 * from current step.
 * \param[in]  contStep Continuation step size.
 */
void
fixedPointCont::continueStep(const double contStep)
{
  if (verbose)
    std::cout << "Continuation step..." << std::endl;
  
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
  predict(S);

  // Apply prediction
  applyPredict(contStep, current);

  // Update Jacobian matrix (the prediction has been applied)
  setCurrentState();
  
  // Correct using Newton-Raphson
  correct();
  
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
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;

  return;
}

/**
 * Get initial state of tracking.
 * If initial_->size == dim + 1, return x(0), T
 * else if initial_->size == dim*numShoot + 1, return x(0), ..., x(numShoot-1), T
 * \param[in]  initial_ Vector in which to copy the initial state.
 */
void
periodicOrbitTrack::getInitialState(gsl_vector *initial_)
{
  if (initial_->size == dim + 1)
    {
      // Set state x(0) (dim element will be changed)
      gsl_vector_view vView = gsl_vector_subvector(initial, 0, dim+1);
      gsl_vector_memcpy(initial_, &vView.vector);
      // Set the period
      gsl_vector_set(initial_, dim, gsl_vector_get(initial, dim * numShoot));
    }
  else if (initial_->size == dim * numShoot + 1)
    gsl_vector_memcpy(initial_, initial);
  else
    std::cerr << "Destination vector size " << initial_->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;

  return;
}

/**
 * Set current state of the model and fundamental matrix
 */
void
periodicOrbitTrack::setCurrentState()
{
  if (verbose)
    std::cout << "Setting current state..." << std::endl;
  
  // Set state of linearized model, i.e. the first dim entries of current,
  // even when multiple shooting.
  gsl_vector_const_view currentState
    = gsl_vector_const_subvector(current, 0, dim);
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
	  linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot,
								     s), dt);
	  currentState = gsl_vector_subvector(current, (s+1)*dim, dim);
	  linMod->mod->getCurrentState(&currentState.vector);
	}
    }
  else if (init->size == dim * numShoot + 1)
    gsl_vector_memcpy(current, init);
  else {
    std::cerr << "Initial state size " << init->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;
    throw std::exception();
  }

  // Set current state of model
  setCurrentState();

  return;
}


/**
 * Set initial state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
periodicOrbitTrack::setInitialState(const gsl_vector *init)
{
  gsl_vector_view initialState;
  
  // Set initial state
  // Check if initCont as size dim+1 or dim*numShoot+1
  if (init->size == dim + 1)
    {
      // Set period
      gsl_vector_set(initial, dim*numShoot, gsl_vector_get(init, dim));
      gsl_vector_set(current, dim*numShoot, gsl_vector_get(init, dim));

      // Adapt time
      adaptTimeToPeriod();
      
      // Set x(0) and initialize model
      initialState = gsl_vector_subvector(initial, 0, dim);
      gsl_vector_const_view vView = gsl_vector_const_subvector(init, 0, dim);
      gsl_vector_memcpy(&initialState.vector, &vView.vector);
      linMod->setCurrentState(&vView.vector);

      // Only x(0) has been given, integrate to x(s) and set
      for (size_t s = 0; s < numShoot - 1; s++)
	{
	  linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot,
								     s), dt);
	  initialState = gsl_vector_subvector(initial, (s+1)*dim, dim);
	  linMod->mod->getCurrentState(&initialState.vector);
	}
    }
  else if (init->size == dim * numShoot + 1)
    gsl_vector_memcpy(initial, init);
  else {
    std::cerr << "Initial state size " << init->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;
    throw std::exception();
  }

  // Set current state of model
  initialize();

  return;
}


/**
 * Initialize.
*/
void
periodicOrbitTrack::initialize()
{
  // Set current state of model
  gsl_vector_memcpy(current, initial);
  setCurrentState();

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
  // Check that period is positive
  if (gsl_vector_get(current, (dim-1) * numShoot + 1) <= 0.) {
    std::cerr << "\nError: period is non-positive." << std::endl;
    converged = false;
    throw std::exception();
  }

  // Get total number of time steps
  nt = (size_t) (ceil(T / intStepCorr) + 0.1);
  
  // Get time step
  dt = T / nt;
  
  // Uniformly divide number of time steps
  gsl_vector_uint_set_all(ntShoot, (size_t) (nt / numShoot));
  
  // Add remaining
  gsl_vector_uint_set(ntShoot, numShoot-1,
		      gsl_vector_uint_get(ntShoot, numShoot-1)
		      + (nt % numShoot));

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
  linMod->integrate(nt, dt);

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
      linMod->integrate((size_t) gsl_vector_uint_get(ntShoot, s), dt);

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

  if (verbose)
    std::cout << "Updating target..." << std::endl;
  
  for (size_t s = 0; s < numShoot; s++)
    {
      // Get a view on x(s+1)
      currentState
	= gsl_vector_subvector(current, ((s+1) % numShoot)*dim, dim);
      // Get a view on part of target (s)
      targetCorrState = gsl_vector_subvector(targetCorr, s*dim, dim);
      // Copy the current state x(s+1) there
      gsl_vector_memcpy(&targetCorrState.vector, &currentState.vector);
  
      /** Integrate model for a period from the new current state
	  after initializing the linear model to x(s)
       *  (no need to update the linear model here). */
      currentState = gsl_vector_subvector(current, s*dim, dim);
      linMod->setCurrentState(&currentState.vector);
      linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot, s),
				    dt);
      std::cout << "integrate for nt = " << gsl_vector_uint_get(ntShoot, s)
		<< "with dt = " << dt << std::endl;

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
  if (verbose)
    std::cout << "Looking for solution..." << std::endl;
  
  numIter = 0;
  
  // Initialize the current state of tracking and that of the linearized model.
  setInitialState(init);

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

      // Check if step in period is not unrealistically large
      if (fabs(gsl_vector_get(stepCorr, dim * numShoot))
	  > gsl_vector_get(current, dim * numShoot)) {
	std::cerr << "\nError: correction step of period "
		  << gsl_vector_get(stepCorr, dim * numShoot)
		  << " larger than previous period "
		  << gsl_vector_get(current, dim * numShoot) << std::endl;
	converged = false;
	throw std::exception();
      }

      // Update correction step size
      errStepCorrSize = gsl_vector_get_norm(stepCorr) / numShoot;

      // Update model state
      applyCorr();

      // Set time step
      adaptTimeToPeriod();

      // Get targetCorr vector to x - \phi_T(x) and state of linearized model
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
      gsl_vector_set(current_, dim-1,
		     gsl_vector_get(current, (dim-1)*numShoot));
      
      // Set the period
      gsl_vector_set(current_, dim,
		     gsl_vector_get(current, (dim-1) * numShoot + 1));
    }
  else if (current_->size == dim * numShoot + 1)
    gsl_vector_memcpy(current_, current);
  else
    std::cerr << "Destination vector size " << current_->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;

  return;
}

/**
 * Set current state x(s), lambda of the model and fundamental matrix
 * \param[in]  s Shooting number
 */
void
periodicOrbitCont::setCurrentState(const size_t s)
{
  if (verbose)
    std::cout << "Setting current state..." << std::endl;
  
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
      
      // Set x(0), lambda and initialize model
      // (last element will be overwritten)
      currentState = gsl_vector_subvector(current, 0, dim);
      gsl_vector_const_view vView = gsl_vector_const_subvector(init, 0, dim);
      gsl_vector_memcpy(&currentState.vector, &vView.vector);
      linMod->setCurrentState(&vView.vector);

      // Only x(0) has been given, integrate to x(s) and set
      for (size_t s = 0; s < numShoot - 1; s++)
	{
	  linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot,
								     s), dt);
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
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;
  
  // Set current state of model
  setCurrentState();

  return;
}


/**
 * Get initial state of tracking.
 * If initial_->size == dim + 1, return x(0), T
 * else if initial_->size == dim*numShoot + 1, return x(0), ..., x(numShoot-1), T
 * \param[in]  initial_ Vector in which to copy the initial state.
 */
void
periodicOrbitCont::getInitialState(gsl_vector *initial_)
{
  if (initial_->size == dim + 1)
    {
      // Set state x(0)
      gsl_vector_view vView = gsl_vector_subvector(initial, 0, dim);
      gsl_vector_view vView2 = gsl_vector_subvector(initial_, 0, dim);
      gsl_vector_memcpy(&vView2.vector, &vView.vector);

      // Set parameter
      gsl_vector_set(initial_, dim-1,
		     gsl_vector_get(initial, (dim-1)*numShoot));
      
      // Set the period
      gsl_vector_set(initial_, dim,
		     gsl_vector_get(initial, (dim-1) * numShoot + 1));
    }
  else if (initial_->size == dim * numShoot + 1)
    gsl_vector_memcpy(initial_, initial);
  else
    std::cerr << "Destination vector size " << initial_->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;

  return;
}

/**
 * Set initial state of the problem together with that of the model.
 * \param[in]  init Initial state.
 */
void
periodicOrbitCont::setInitialState(const gsl_vector *init)
{
  gsl_vector_view initialState;
  
  // Set initial state
  // Check if initCont as size dim+1 or dim*numShoot+1
  if (init->size == dim + 1)
    {
      // Set period (to adapt time)
      gsl_vector_set(initial, (dim-1)*numShoot+1, gsl_vector_get(init, dim));
      gsl_vector_set(current, (dim-1)*numShoot+1, gsl_vector_get(init, dim));

      // Adapt time
      adaptTimeToPeriod();
      
      // Set x(0), lambda and initialize model
      // (last element will be overwritten)
      initialState = gsl_vector_subvector(initial, 0, dim);
      gsl_vector_const_view vView = gsl_vector_const_subvector(init, 0, dim);
      gsl_vector_memcpy(&initialState.vector, &vView.vector);
      linMod->setCurrentState(&vView.vector);

      // Only x(0) has been given, integrate to x(s) and set
      for (size_t s = 0; s < numShoot - 1; s++)
	{
	  linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot,
								     s), dt);
	  initialState = gsl_vector_subvector(initial, (s+1)*(dim-1), dim);
	  linMod->mod->getCurrentState(&initialState.vector);
	}
      
      // Set period (again, because overwritten)
      gsl_vector_set(initial, (dim-1)*numShoot+1, gsl_vector_get(init, dim));
    }
  else if (init->size == (dim-1) * numShoot + 2)
    gsl_vector_memcpy(initial, init);
  else
    std::cerr << "Initial state size " << init->size
	      << " does not match dim + 1 = " << (dim + 1)
	      << " nor dim * numShoot + 1 = " << (dim * numShoot + 1)
	      << std::endl;
  
  // Set current state of model
  initialize();

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
      // Add F(x(s)) to last row (the last entry will be updated,
      // with the very last 0)
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
      linMod->integrate((size_t) gsl_vector_uint_get(ntShoot, s), dt);

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

  if (verbose)
    std::cout << "Updating target..." << std::endl;
  
  for (size_t s = 0; s < numShoot; s++)
    {
      // Get a view on part of target (s)
      // (the last element will be overwritten)
      targetCorrState = gsl_vector_subvector(targetCorr, s*(dim-1), dim);
      
      // Get extended state vector x(s+1), lambda and copy it to target (s)
      getExtendedState(work, ((s + 1) % numShoot));
      gsl_vector_memcpy(&targetCorrState.vector, work);
  
      /** Integrate model for a period from the new current state
	  after initializing the linear model to x(s)
       *  (no need to update the linear model here). */
      setCurrentState(s);
      linMod->mod->integrate((size_t) gsl_vector_uint_get(ntShoot,
							  s), dt);

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
 * Correct prediction by pseudo-arclenght continuation and Newton-Raphson.
 */
void
periodicOrbitCont::correct()
{
  if (verbose)
    std::cout << "Correcting..." << std::endl;
  
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
	  std::cout << "Correction iteration " << numIter
		    << " with previous step:\n"
		    << "Current state: " << std::endl;
	  gsl_vector_fprintf(stdout, current, "%lf");
	  std::cout << "Correction step: " << std::endl;
	  gsl_vector_fprintf(stdout, stepCorr, "%lf");
	  std::cout << "errDist: " << errDist << "\nerrStepCorrSize = "
		    << errStepCorrSize << std::endl;
	}
      
      // Perform Newton step
      NewtonStep();
      
      // Check if step in period is not unrealistically large
      if (fabs(gsl_vector_get(stepCorr, dim * numShoot))
	  > gsl_vector_get(current, dim * numShoot)) {
	std::cerr << "\nError: correction step of period "
		  << gsl_vector_get(stepCorr, dim * numShoot)
		  << " larger than previous period "
		  << gsl_vector_get(current, dim * numShoot) << std::endl;
	converged = false;
	throw std::exception();
      }

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

      if (verbose)
	std::cout << "" << std::endl;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

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
  
  if (verbose)
    std::cout << "Continuation step..." << std::endl;
  
  // Initialize to current state
  setCurrentState();

  // Previous prediction
  if (verbose)
    {
      std::cout << "Previous prediction step:" << std::endl;
      gsl_vector_fprintf(stdout, stepPred, "%lf");
    }

  // Predict
  predict(S);

  // Check if step in period is not unrealistically large
  if (fabs(gsl_vector_get(stepPred, dim * numShoot) * contStep)
      > gsl_vector_get(current, dim * numShoot)) {
    std::cerr << "\nError: prediction step of period "
	      << (gsl_vector_get(stepPred, dim * numShoot) * contStep)
	      << " larger than previous period "
	      << gsl_vector_get(current, dim * numShoot) << std::endl;
    converged = false;
    throw std::exception();
  }

  // Apply prediction
  applyPredict(contStep, current);

  // Update model and Jacobian to current state
  // (the prediction has been applied)
  setCurrentState();
  
  // Correct using Newton-Raphson
  if (verbose)
    std::cout << "Correcting..." << std::endl;
  correct();
  
  return;
}


/**
 * Calculate covariance matrix \int_0^T M(T, r) * Q(r) * M(T, r).T dt
 * from time series of fundamental matrices M(t, s)
 * and of diffusion matrices Q(s), 0 <= s < nt, with T = nt * dt.
 * \param[in]  Qs  Time series of diffusion matrices.
 * \param[in]  Mts Time series of fundamental matrices.
 * \param[out] CT  Covariance matrix.
 * \param[in]  dt  Time step.
 */
void
getCovarianceMatrix(const std::vector<gsl_matrix *> *Qs,
		    const std::vector<gsl_matrix *> *Mts,
		    gsl_matrix *CT, const double dt)
{
  const size_t nt = Mts->size() - 1;
  size_t dim = CT->size1;
  gsl_matrix *tmp = gsl_matrix_alloc(dim, dim);
  gsl_matrix *tmp2 = gsl_matrix_alloc(dim, dim);

  // Calculate int_0^T M(t, r) * Q(r) * M(t, r).T dr
  gsl_matrix_set_zero(CT);
  for (size_t r = 0; r < nt; r++) {
    // Q(r) * M(t, r).T
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., Qs->at(r), Mts->at(r),
		   0., tmp2);
    // M(t, r) * Q(r) * M(t, r).T
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., Mts->at(r), tmp2,
		   0., tmp);

    // Add the result to the correlation matrix
    gsl_matrix_add(CT, tmp);
  }

  // Scale with time
  gsl_matrix_scale(CT, dt);

  // Free
  gsl_matrix_free(tmp2);
  gsl_matrix_free(tmp);

  return;
}

  
/**
 * Calculate phase diffusion coefficient from covariance matrix C(T, 0).
 * \param[in]  CT     Covariance matrix.
 * \param[in]  vLeft  Left vector on which to project.
 * \param[in]  vRight Right vector on which to project.
 * \param[in]  T      Period of integration.
 * \return            Phase diffusion coefficient
 */
double
getPhaseDiffusion(const gsl_matrix *CT,
		  const gsl_vector *vLeft, const gsl_vector *vRight,
		  const double T)
{
  double phi;
  size_t dim = vLeft->size;
  gsl_vector *CTvLeft = gsl_vector_alloc(dim);

  // Get C(T, 0) * vLeft
  gsl_blas_dgemv(CblasNoTrans, 1., CT, vLeft, 0., CTvLeft);

  // Get vLeft * C(T, 0) * vLeft
  phi = gsl_vector_get_inner_product(vLeft, CTvLeft)
    / gsl_vector_get_inner_product(vLeft, vRight) / T;

  // Free
  gsl_vector_free(CTvLeft);
  
  return phi;
}


/**
 * Calculate phase diffusion coefficient
 * from time series of fundamental matrices M(t, s)
 * and of diffusion matrices Q(s), 0 <= s < nt, with T = nt * dt.
 * \param[in]  Qs     Time series of diffusion matrices.
 * \param[in]  Mts    Time series of fundamental matrices.
 * \param[in]  vLeft  Left vector on which to project.
 * \param[in]  vRight Right vector on which to project.
 * \param[in]  dt     Time step of integration.
 * \return            Phase diffusion coefficient
 */
double
getPhaseDiffusion(const std::vector<gsl_matrix *> *Qs,
		  const std::vector<gsl_matrix *> *Mts,
		  const gsl_vector *vLeft, const gsl_vector *vRight,
		  const double dt)
{
  double phi;
  const size_t dim = vLeft->size;
  const size_t nt = Mts->size() - 1;
  const double T = dt * nt;
  gsl_matrix *CT = gsl_matrix_alloc(dim, dim);

  // Get covariance matrix
  getCovarianceMatrix(Qs, Mts, CT, dt);

  // Get phase diffusion
  phi = getPhaseDiffusion(CT, vLeft, vRight, T);

  gsl_matrix_free(CT);

  return phi;
}


/**
 * Get Floquet eigenvalues and left and right eigenvectors from the
 * current state of the periodic orbit continuation.
 * \param[in]  track       Periodic ontinuation problem.
 * \param[out] state       State vector on orbit.
 * \param[out] FloquetExp  Floquet exponents.
 * \param[out] eigVecLeft  Left Floquet vectors.
 * \param[out] eigVecRight Right Floquet vectors.
 * \param[in]  sort        Whether to sort the spectrum or not.
 */
void
getFloquet(periodicOrbitCont *track, gsl_vector *state,
	   gsl_vector_complex *FloquetExp,
	   gsl_matrix_complex *eigVecLeft, gsl_matrix_complex *eigVecRight,
	   const bool sort)
{
  const size_t dim = track->getDim() - 1;
  double T;
  gsl_matrix_view fm;
  gsl_vector_complex *eigValLeft = gsl_vector_complex_alloc(dim);
  gsl_vector_complex *eigValRight = gsl_vector_complex_alloc(dim);
  gsl_matrix *trans = gsl_matrix_alloc(dim, dim);
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(dim);
  gsl_matrix *solFM = gsl_matrix_alloc(dim + 1, dim + 1);
  
  // Get period and the fundamental matrix
  track->getCurrentState(state);
  T = gsl_vector_get(state, dim + 1);
  track->getStabilityMatrix(solFM);
  fm = gsl_matrix_submatrix(solFM, 0, 0, dim, dim);

  // Find eigenvalues left
  gsl_matrix_transpose_memcpy(trans, &fm.matrix);
  gsl_eigen_nonsymmv(trans, eigValLeft, eigVecLeft, w);
  
  // Find eigenvalues right
  gsl_eigen_nonsymmv(&fm.matrix, eigValRight, eigVecRight, w);

  // Sort
  if (sort)
    sortSpectrum(eigValLeft, eigVecLeft, eigValRight, eigVecRight);

  // Convert to Floquet exponents
  gsl_vector_complex_log(FloquetExp, eigValLeft);
  gsl_vector_complex_scale_real(FloquetExp, 1. / T);

  // Free
  gsl_matrix_free(solFM);
  gsl_matrix_free(trans);
  gsl_eigen_nonsymmv_free(w);
  gsl_vector_complex_free(eigValLeft);
  gsl_vector_complex_free(eigValRight);

  return;
}


/**
 * Write Floquet eigenvalues and left and right eigenvectors.
 * \param[in]  track          Periodic ontinuation problem.
 * \param[in]  state          State vector on orbit.
 * \param[in]  FloquetExp     Floquet exponents.
 * \param[in]  eigVecLeft     Left Floquet vectors.
 * \param[in]  eigVecRight    Right Floquet vectors.
 * \param[out] streamState    Stream in which to write state.
 * \param[out] streamExp      Stream in which to write Floquet exponents.
 * \param[out] streamVecLeft  Stream in which to write left Floquet vectors.
 * \param[out] streamVecRight Stream in which to write right Floquet vectors.
 * \param[in]  verbose        Print results to standard output.
 */
void
writeFloquet(periodicOrbitCont *track, const gsl_vector *state,
	     const gsl_vector_complex *FloquetExp,
	     const gsl_matrix_complex *eigVecLeft,
	     const gsl_matrix_complex *eigVecRight,
	     FILE *streamState, FILE *streamExp,
	     FILE *streamVecLeft, FILE *streamVecRight,
	     const char *fileFormat, const bool verbose)
{
  // Print periodic orbit
  if (verbose) {
    std::cout << "periodic orbit:" << std::endl;
    gsl_vector_fprintf(stdout, state, "%lf");
    std::cout << "Eigenvalues:" << std::endl;
    gsl_vector_complex_fprintf(stdout, FloquetExp, "%lf");
  }

  // Write results
  if (strcmp(fileFormat, "bin") == 0)
    {
      gsl_vector_fwrite(streamState, state);
      gsl_vector_complex_fwrite(streamExp, FloquetExp);
      gsl_matrix_complex_fwrite(streamVecLeft, eigVecLeft);
      gsl_matrix_complex_fwrite(streamVecRight, eigVecRight);
    }
  else
    {
      gsl_vector_fprintf(streamState, state, "%lf");
      gsl_vector_complex_fprintf(streamExp, FloquetExp, "%lf");
      gsl_matrix_complex_fprintf(streamVecLeft, eigVecLeft, "%lf");
      gsl_matrix_complex_fprintf(streamVecRight, eigVecRight, "%lf");
    }

  return;
}


/**
 * Sort by left and right eigenvalues and eigenvectors
 * by largest largest maginitude of eigenvalue.
 * \param[in/out]  eigValLeft Left eigenvalues.
 * \param[in/out]  eigVecLeft Left eigenvectors.
 * \param[in/out]  eigValRight Right eigenvalues.
 * \param[in/out]  eigVecRight Right eigenvectors.
 */
void
sortSpectrum(gsl_vector_complex *eigValLeft, gsl_matrix_complex *eigVecLeft,
	     gsl_vector_complex *eigValRight, gsl_matrix_complex *eigVecRight)
{
  const size_t dim = eigValLeft->size;
  gsl_vector_complex *tmpValRight = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *tmpVecRight = gsl_matrix_complex_alloc(dim, dim);
  gsl_vector_complex *tmpValLeft = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *tmpVecLeft = gsl_matrix_complex_alloc(dim, dim);
  gsl_permutation *sort_idx = gsl_permutation_alloc(dim);
  gsl_vector *absEigVal;
  gsl_complex tmp;
  size_t idx;

  // Sort right eigenvalues and vectors
  absEigVal = gsl_vector_complex_abs(eigValRight);
  gsl_sort_vector_index(sort_idx, absEigVal);
  gsl_permutation_reverse(sort_idx);
  gsl_permute_vector_complex(sort_idx, eigValRight);
  gsl_permute_matrix_complex(sort_idx, eigVecRight, 1);

  /** Sort left vectors by correspondance to for counterparts
   *  (since different eigenvalues may have the same magnitude). */
  gsl_matrix_complex_memcpy(tmpVecLeft, eigVecLeft);
  gsl_vector_complex_memcpy(tmpValLeft, eigValLeft);
  for (size_t ev = 0; ev < dim; ev++)
    {
      //! Get distance from eigenvalue
      gsl_vector_complex_memcpy(tmpValRight, eigValRight);
      tmp = gsl_complex_negative(gsl_complex_conjugate(gsl_vector_complex_get(tmpValLeft, ev)));
      gsl_vector_complex_add_constant(tmpValRight, tmp);
      idx = gsl_vector_complex_min_index(tmpValRight);

      //! Sort left eigenvalue and eigenvector
      gsl_vector_complex_set(eigValLeft, idx,
			     gsl_vector_complex_get(tmpValLeft, ev));
      gsl_vector_complex_const_view view =
	gsl_matrix_complex_const_row(tmpVecLeft, ev);
      gsl_matrix_complex_set_row(eigVecLeft, idx, &view.vector);
    }

  //! Free
  gsl_vector_complex_free(tmpValRight);
  gsl_matrix_complex_free(tmpVecRight);
  gsl_vector_complex_free(tmpValLeft);
  gsl_matrix_complex_free(tmpVecLeft);
  gsl_vector_free(absEigVal);
  gsl_permutation_free(sort_idx);
  
  return;
}
