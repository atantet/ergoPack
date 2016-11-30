#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl_extension.hpp>
#include <math.h>
#include <iostream>


/**
 * Get current state of tracking.
 * \param[in]  current_ Vector in which to copy the current state.
 */
void
solutionTrack::getCurrentState(gsl_vector *current_)
{
  gsl_vector_memcpy(current_, current);

  return;
}

/** 
 * Update state after Newton step.
 * \param[in] damping Damping to apply to the Newton step.
 */
void
solutionTrack::applyStep(const double damping)
{
  // Damp step
  gsl_vector_scale(step, damping);
  
  // Update current state
  gsl_vector_add(current, step);
  
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

  // Set state of linearized model
  linMod->setCurrentState(current);

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
  linMod->Jacobian->getParameters(matrix);
  
  return;
}

/**
 * Perform a Newton step to find a fixed point.
 */
void
fixedPointTrack::NewtonStep()
{
  gsl_permutation *p = gsl_permutation_alloc(dim);
  int s;

  // Set up LU decomposition
  // Note that this modifies the matrix A
  gsl_linalg_LU_decomp(linMod->Jacobian->A, p, &s);

  // Solve the linear system
  gsl_linalg_LU_solve(linMod->Jacobian->A, p, target, step);

  return;
}


/**
 * Update target vector for fixed point tracking.
 */
void
fixedPointTrack::updateTarget()
{
  // Update target
  linMod->mod->field->evalField(current, target);
  gsl_vector_scale(target, -1.);

  return;
}


/**
 * Find fixed point using Newton-Raphson method.
 * \param[in]  init Vector from which to start tracking.
 */
void
fixedPointTrack::findSolution(const gsl_vector *init)
{
  numIter = 0;
  
  // Initialize current state of tracking and linearized model state
  setCurrentState(init);  
    
  // Get target vector -f(x)
  updateTarget();
  
  while (((errDist > epsDist) || (errStepSize > epsStepSize)) && (numIter < maxIter))
    {
      // Perform Newton step
      NewtonStep();

      // Update step size before to damp it
      errStepSize = sqrt(gsl_vector_get_sum_squares(step));

      // Update model state
      applyStep();

      // Get target vector -f(x)
      updateTarget();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      linMod->setCurrentState(current);

      // Update distance to target and iterate
      errDist = sqrt(gsl_vector_get_sum_squares(target));
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;

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

  // Set state of linearized model
  gsl_vector_const_view initState = gsl_vector_const_subvector(current, 0, dim);
  linMod->setCurrentState(&initState.vector);

  return;
}


/**
 * Get fundamental matrix of the current state.
 * \param[in]  Matrix in which to copy the fundamental matrix.
 */
void
periodicOrbitTrack::getStabilityMatrix(gsl_matrix *matrix)
{
  // Get integration time step.
  const double T = gsl_vector_get(current, dim);
  nt = (int) (ceil(T / intStep) + 0.1);
  dt = T / nt;

  // Get solution and fundamental matrix after the period
  linMod->integrateForward(nt, dt);

  // Copy
  linMod->getCurrentState(matrix);
  
  // Rewind
  linMod->setCurrentState(&currentState.vector);
  
  return;
}


/**
 * Get matrix of the linear system to be solved.
 */
void
periodicOrbitTrack::getLinearSystem()
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
  linMod->mod->field->evalField(linMod->mod->current, &vView.vector);

  // Get solution and fundamental matrix after the period
  linMod->integrateForward(nt, dt);

  // Set data after a period
  mView = gsl_matrix_submatrix(S, 0, 0, dim, dim);
  // Add MT to state part
  gsl_matrix_add(&mView.matrix, linMod->current);
  // Add F(xT) to upper right corner
  vView = gsl_matrix_subcolumn(S, dim, 0, dim);
  linMod->mod->field->evalField(linMod->mod->current, &vView.vector);
    
  return;
}

/**
 * Perform a Newton step to find a periodic orbit.
 */
void
periodicOrbitTrack::NewtonStep()
{
  gsl_permutation *p = gsl_permutation_alloc(dim + 1);
  int s;

  // Get linear system
  getLinearSystem();

  // Set up LU decomposition
  // Note that this modifies the matrix A
  gsl_linalg_LU_decomp(S, p, &s);

  // Solve the linear system to get new step
  gsl_linalg_LU_solve(S, p, target, step);

  return;
}


/**
 * Update target vector for periodic orbit tracking.
 */
void
periodicOrbitTrack::updateTarget()
{
  // Creatte views on the state part of the target and the current state
  gsl_vector_view targetState = gsl_vector_subvector(target, 0, dim);

  // Copy the current state there
  gsl_vector_memcpy(target, current);
  
  /** Integrate model for a period from the new current state
   *  (no need to update the linear model here). */
  linMod->mod->integrateForward(&currentState.vector, nt, dt);

  // Substract the new state
  gsl_vector_sub(&targetState.vector, linMod->mod->current);

  // Set target for period to 0
  gsl_vector_set(target, dim, 0.);

  return;
}


/**
 * Find periodic orbit using Newton-Raphson method.
 * \param[in]  init Vector from which to start tracking.
 */
void
periodicOrbitTrack::findSolution(const gsl_vector *init)
{
  numIter = 0;
  double T;
  
  // Initialize the current state of trakcing and that of the linearized model.
  setCurrentState(init);
    
  // Get integration time step.
  T = gsl_vector_get(current, dim);
  nt = (int) (ceil(T / intStep) + 0.1);
  dt = T / nt;

  // Get target vector to x - \phi_T(x) and update error
  updateTarget();
  errDist = sqrt(gsl_vector_get_sum_squares(target));

  /** Update state x of model and Jacobian J(x) to current state
   *  and reinitizlize fundamental matrix to identity
   *  (after integration in updateTarget). */
  linMod->setCurrentState(&currentState.vector);
    
  while (((errDist > epsDist) || (errStepSize > epsStepSize)) && (numIter < maxIter))
    {
      std::cout << "numIter = " << numIter << std::endl;
      std::cout << "errDist = " << errDist << std::endl;
      std::cout << "errStepSize = " << errStepSize << std::endl;
      // Perform Newton step (leaves the model integrated forward)
      NewtonStep();

      // Update step size before to damp it
      errStepSize = sqrt(gsl_vector_get_sum_squares(step));

      // Update model state
      applyStep();

      // Get integration time step from new current state.
      T = gsl_vector_get(current, dim);
      nt = (int) (ceil(T / intStep) + 0.1);
      dt = T / nt;

      // Get target vector to x - \phi_T(x) and state of the linearized model
      updateTarget();

      /** Update state x of model and Jacobian J(x) to current state
       *  and reinitizlize fundamental matrix to identity. */
      linMod->setCurrentState(&currentState.vector);
    
      // Update distance to target
      errDist = sqrt(gsl_vector_get_sum_squares(target));
      numIter++;
    }

  /** Update the convergence flag. */
  if (numIter < maxIter)
    converged = true;
  
  return;
  
}


// /**
//  * Prediction Newton-Raphson step.
//  */
// void
// fixedPointCont::predict()
// {
//   gsl_permutation *p = gsl_permutation_alloc(dim + 1);
//   int s;

//   // Get linear system
//   getLinearSystem();

//   // Set up LU decomposition
//   // Note that this modifies the matrix A
//   gsl_linalg_LU_decomp(S, p, &s);

//   // Solve the linear system to get new step
//   gsl_linalg_LU_solve(S, p, target, step);

//   return;
// }


// /**
//  * Get matrix of the linear system to be solved.
//  */
// void
// fixedPointCont::getLinearSystem()
// {
//   gsl_matrix_view mView;
//   gsl_vector_view vView;

//   // Set initial data
//   // View on the upper left corner
//   mView = gsl_matrix_submatrix(S, 0, 0, dim, dim + 1);
//   // Set to minus the current matrix of linear model, i.e., the identity by default
//   linMod->getCurrentState(&mView.matrix);
//   gsl_matrix_scale(&mView.matrix, -1.);
//   // Add F(x0) to lower left corner
//   vView = gsl_matrix_subrow(S, dim, 0, dim);
//   linMod->mod->field->evalField(linMod->mod->current, &vView.vector);

//   // Get solution and fundamental matrix after the period
//   linMod->integrateForward(nt, dt);

//   // Set data after a period
//   mView = gsl_matrix_submatrix(S, 0, 0, dim, dim);
//   // Add MT to state part
//   gsl_matrix_add(&mView.matrix, linMod->current);
//   // Add F(xT) to upper right corner
//   vView = gsl_matrix_subcolumn(S, dim, 0, dim);
//   linMod->mod->field->evalField(linMod->mod->current, &vView.vector);
    
//   return;
// }

