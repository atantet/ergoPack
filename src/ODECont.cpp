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
      errStepCorrSize = sqrt(gsl_vector_get_sum_squares(stepCorr));

      // Update model state
      applyCorr();

      // Get targetCorr vector -f(x)
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state */
      setCurrentState();

      // Update distance to targetCorr and iterate
      errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));
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
      errStepCorrSize = sqrt(gsl_vector_get_sum_squares(stepCorr));

      // Update model state
      applyCorr();

      // Get target vector -f(x)
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state */
      setCurrentState();

      // Update distance to target and iterate
      errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));
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
 * Set current state of the model and fundamental matrix
 */
void
periodicOrbitTrack::setCurrentState()
{
  // Set state of linearized model
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
  // Set current state
  gsl_vector_memcpy(current, init);

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
  // Get integration time stepCorr.
  const double T = gsl_vector_get(current, dim);
  nt = (int) (ceil(T / intStepCorr) + 0.1);
  dt = T / nt;

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
  gsl_vector_view vView;

  // Set initial data
  gsl_matrix_set(S, dim, dim, 0.);
  // View on the upper left corner
  mView = gsl_matrix_submatrix(S, 0, 0, dim, dim);
  // Set to minus the current matrix of linear model, i.e., the identity by default
  linMod->getCurrentState(&mView.matrix);
  gsl_matrix_scale(&mView.matrix, -1.);
  // Add F(x0) to lower left corner
  vView = gsl_matrix_subrow(S, dim, 0, dim);
  linMod->mod->evalField(linMod->mod->current, &vView.vector);

  // Get solution and fundamental matrix after the period
  linMod->integrateForward(nt, dt);

  // Set data after a period
  mView = gsl_matrix_submatrix(S, 0, 0, dim, dim);
  // Add MT to state part
  gsl_matrix_add(&mView.matrix, linMod->current);
  // Add F(xT) to upper right corner
  vView = gsl_matrix_subcolumn(S, dim, 0, dim);
  linMod->mod->evalField(linMod->mod->current, &vView.vector);
    
  return;
}


/**
 * Update targetCorr vector for periodic orbit tracking.
 */
void
periodicOrbitCorr::updateTargetCorr()
{
  // Creatte views on the state part of the targetCorr and the current state
  gsl_vector_view targetCorrState = gsl_vector_subvector(targetCorr, 0, dim);
  gsl_vector_view currentState = gsl_vector_subvector(current, 0, dim);

  // Copy the current state there
  gsl_vector_memcpy(targetCorr, current);
  
  /** Integrate model for a period from the new current state
   *  (no need to update the linear model here). */
  linMod->mod->integrateForward(&currentState.vector, nt, dt);

  // Substract the new state
  gsl_vector_sub(&targetCorrState.vector, linMod->mod->current);

  // Set targetCorr for period to 0
  gsl_vector_set(targetCorr, dim, 0.);

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
  double T;
  
  // Initialize the current state of tracking and that of the linearized model.
  setCurrentState(init);
    
  // Get integration time stepCorr.
  T = gsl_vector_get(current, dim);
  nt = (int) (ceil(T / intStepCorr) + 0.1);
  dt = T / nt;

  // Get targetCorr vector to x - \phi_T(x) and update error
  updateTargetCorr();
  errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));

  /** Update state x of model and Jacobian J(x) to current state
   *  and reinitizlize fundamental matrix to identity
   *  (after integration in updateTargetCorr). */
  setCurrentState();
    
  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      // Perform Newton step (leaves the model integrated forward)
      NewtonStep();

      // Update correction step size before to damp it
      errStepCorrSize = sqrt(gsl_vector_get_sum_squares(stepCorr));

      // Update model state
      applyCorr();

      // Get integration time stepCorr from new current state.
      T = gsl_vector_get(current, dim);
      nt = (int) (ceil(T / intStepCorr) + 0.1);
      dt = T / nt;

      // Get targetCorr vector to x - \phi_T(x) and state of the linearized model
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      setCurrentState();
    
      // Update distance to targetCorr
      errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));
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
 * Get matrix of the linear system to be solved for correction.
 */
void
periodicOrbitCont::getSystemCorr()
{
  gsl_matrix_view mView, mView2;
  gsl_vector_view vView;

  // Set initial data
  // View on the upper left corner
  mView = gsl_matrix_submatrix(S, 0, 0, dim, dim);
  // Set to minus the current matrix of linear model, i.e., the identity by default
  // including the last row/col for the parameter.
  // S[:dim+1, :dim+1] = -I, S[dim, :dim+1] will be changed
  linMod->getCurrentState(&mView.matrix);
  gsl_matrix_scale(&mView.matrix, -1.);
  // Add F(x0) to lower left corner (the last entry is 0)
  // S[dim+1, :dim+1] = F(x) with F[dim] = 0.
  vView = gsl_matrix_subrow(S, dim, 0, dim);
  linMod->mod->evalField(linMod->mod->current, &vView.vector);

  // Get solution and fundamental matrix after the period
  linMod->integrateForward(nt, dt);

  // Set data after a period
  // Add MT to state part, including the last column for parameter
  // S[:dim, :dim+1] += M(t), ddalpha
  mView = gsl_matrix_submatrix(S, 0, 0, dim-1, dim);
  mView2 = gsl_matrix_submatrix(linMod->current, 0, 0, dim-1, dim);
  gsl_matrix_add(&mView.matrix, &mView2.matrix);
  // Add F(xT) to upper right corner
  // S[:dim+1, dim+1] = F(xT).T
  vView = gsl_matrix_subcolumn(S, dim, 0, dim);
  linMod->mod->evalField(linMod->mod->current, &vView.vector);

  // Set the middle part to the previous prediction step
  // to normalize the step.
  // S[dim+1, :dim+1] = v
  vView = gsl_matrix_subrow(S, dim-1, 0, dim+1);
  gsl_vector_memcpy(&vView.vector, stepPred);
  // Set last element to zero to exclude period from normalization
  gsl_matrix_set(S, dim-1, dim, 0.);

  // Set bottom right corner to zero S[dim+1, dim+1] = 0.
  gsl_matrix_set(S, dim, dim, 0.);

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
  gsl_permutation *p = gsl_permutation_alloc(dim + 1);
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
  gsl_vector_add(current, stepPred);

  // Scale back, since the continuation step is used as later
  gsl_vector_scale(stepPred, 1. / contStep);

  return;
}


/**
 * Update correction target vector for fixed point tracking.
 */
void
periodicOrbitCont::updateTargetCorr()
{
  // Creatte views on the state part of the targetCorr and the current state
  gsl_vector_view targetCorrState = gsl_vector_subvector(targetCorr, 0, dim);
  gsl_vector_view currentState = gsl_vector_subvector(current, 0, dim);

  // Copy the current state there
  gsl_vector_memcpy(targetCorr, current);
  
  /** Integrate model for a period from the new current state
   *  (no need to update the linear model here). */
  linMod->mod->integrateForward(&currentState.vector, nt, dt);

  // Substract the new state
  gsl_vector_sub(&targetCorrState.vector, linMod->mod->current);

  // Set targetCorr for parameter to 0 (pseudo-arclength)
  gsl_vector_set(targetCorr, dim-1, 0.);

  // Set targetCorr for period to 0
  gsl_vector_set(targetCorr, dim, 0.);

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
  double T;
  
  // Get integration time stepCorr.
  T = gsl_vector_get(current, dim);
  nt = (int) (ceil(T / intStepCorr) + 0.1);
  dt = T / nt;

  // Get targetCorr vector to x - \phi_T(x) and update error
  errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));
  updateTargetCorr();

  /** Update state x of model and Jacobian J(x) to current state
   *  and reinitizlize fundamental matrix to identity
   *  (after integration in updateTargetCorr). */
  setCurrentState();
    
  while (((errDist > epsDist) || (errStepCorrSize > epsStepCorrSize))
	 && (numIter < maxIter))
    {
      // Perform Newton step
      NewtonStep();

      // Update step size before to damp it
      errStepCorrSize = sqrt(gsl_vector_get_sum_squares(stepCorr));

      // Update model state
      applyCorr();

      // Get integration time stepCorr from new current state.
      T = gsl_vector_get(current, dim);
      nt = (int) (ceil(T / intStepCorr) + 0.1);
      dt = T / nt;

      // Get target vector
      updateTargetCorr();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      setCurrentState();
    
      // Update distance to target and iterate
      errDist = sqrt(gsl_vector_get_sum_squares(targetCorr));
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

  // Predict
  predict();

  // Apply prediction
  applyPredict(contStep);

  // Update model and Jacobian to current state (the prediction has been applied)
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
periodicOrbitCont::continueStep(const double contStep, const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);

  // Continue
  continueStep(contStep);
  
  return;
}


