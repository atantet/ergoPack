#ifndef ODECONT_HPP
#define ODECONT_HPP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <iostream>

/** \file ODECont.hpp
 *  \brief Continuation of Ordinary Differential Equations.
 *   
 *  Continuation of Ordinary Differential Equations.
 */


/** \ingroup continuation
 * @{
 */

class solutionCorrection {

protected:
  size_t dim;               //!< Dimension of the problem
  const double epsDist;     //!< Tolerance in the distance to zero
  const double epsStepCorrSize; //!< Tolerance in the size of stepCorr
  const size_t maxIter;     //!< Maximum number of iterations
  double errDist;           //!< Distance to zero
  double errStepCorrSize;       //!< Size of stepCorr
  gsl_vector *stepCorr;         //!< StepCorr correcting the solution
  gsl_vector *targetCorr;       //!< TargetCorr vector
  size_t numIter;           //!< Number of iterations
  bool converged;           //!< Flag for convergence of the tracking.
  gsl_matrix *S;            //!< Matrix of the linear system to solve (workspace).
  gsl_vector *current;      //!< Current state of tracking
  bool verbose;           //!< Verbose mode.

  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  solutionCorrection(const double epsDist_, const double epsStepCorrSize_,
		     const size_t maxIter_, const bool verbose_=false)
    : epsDist(epsDist_), epsStepCorrSize(epsStepCorrSize_), maxIter(maxIter_),
      verbose(verbose_),
      errDist(1.e27), errStepCorrSize(1.e27), numIter(0), converged(false) {}

  /** \brief Return the dimension of the problem. */
  size_t getDim() { return dim; }
  
  /** \brief Return the number of iterations used for tracking. */
  size_t getNumIter() { return numIter; }

  /** \brief Return the last distance to the targetCorr. */
  double getDist() { return errDist; }

  /** \brief Return the size of the last stepCorr. */
  double getStepCorrSize() { return errStepCorrSize; }

  /** \brief Return the flag for the convergence of the tracking. */
  bool hasConverged() { return converged; }
  
  /** \brief Perform a Newton step to find a fixed point. */
  void NewtonStep();
  
  /** \brief Correct with the stepCorr obtained from the Newton stepCorr. */
  void applyCorr(const double damping=1);

  /** \brief Destructor. */
  virtual ~solutionCorrection(){};

  /** \brief Get current state of tracking. */
  virtual void getCurrentState(gsl_vector *current_);

  /** \brief Set current state of the models from the current state. */
  virtual void setCurrentState() = 0;

  /** \brief Set current state of the problem together with that of the model. */
  virtual void setCurrentState(const gsl_vector *init) = 0;

  /** \brief Update targetCorr. */
  virtual void updateTargetCorr() = 0;

  /** \brief Get stability matrix associated with the solution. */
  virtual void getStabilityMatrix(gsl_matrix *matrix) = 0;

  /** \brief Get matrix of the linear system to be solved for correction. */
  virtual void getSystemCorr() = 0;

};


class fixedPointTrack : public solutionCorrection {

protected:
  vectorField * const field;    //!< Nonlinear vector field
  linearField * const Jacobian; //!< Linearized vector field
  
public:
  
  /** \brief Constructor assigning a linearized model and parameters. */
  fixedPointTrack(vectorField *field_, linearField *Jac, const double epsDist_,
		  const double epsStepCorrSize_, const size_t maxIter_,
		  const bool verbose_=false)
    : solutionCorrection(epsDist_, epsStepCorrSize_, maxIter_, verbose_),
      field(field_), Jacobian(Jac) { dim = Jac->getRows(); }

  ~fixedPointTrack()
  { gsl_vector_free(current);
    gsl_vector_free(stepCorr);
    gsl_vector_free(targetCorr);
    gsl_matrix_free(S); }

  /** \brief Set current state of model and Jacobian from the current state. */
  void setCurrentState();

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

  /** \brief Update correction target vector for periodic orbit tracking. */
  virtual void updateTargetCorr() = 0;
  
  /** \brief Get the Jacobian of the solution. */
  void getStabilityMatrix(gsl_matrix *matrix);

  /** \brief Get matrix of the linear system to be solved for correction. */
  virtual void getSystemCorr() = 0;
};


class fixedPointCorr : public fixedPointTrack {

public:
  /** \brief Constructor assigning a linearized model and parameters. */
  fixedPointCorr(vectorField *field_, linearField *Jac, const double epsDist_,
		  const double epsStepCorrSize_, const size_t maxIter_,
		  const bool verbose_=false)
    : fixedPointTrack(field_, Jac, epsDist_, epsStepCorrSize_, maxIter_,
		      verbose_) {
    current = gsl_vector_alloc(dim);
    S = gsl_matrix_alloc(dim, dim);
    stepCorr = gsl_vector_calloc(dim);
    targetCorr = gsl_vector_alloc(dim);
  }

  ~fixedPointCorr() { }
  
  /** \brief Find fixed point using Newton-Raphson method. */
  void findSolution(const gsl_vector *init);

  /** \brief Update targetCorr. */
  void updateTargetCorr();

  /** \brief Get matrix of the linear system to be solved for correction. */
  void getSystemCorr();
};


class fixedPointCont : public fixedPointTrack {

private:
  gsl_vector *stepPred;         //!< Step of prediction.
  gsl_vector *targetPred;       //!< (0, ..., 0, 1) vector for prediction

public:
  /** \brief Constructor assigning a linearized model and parameters. */
  fixedPointCont(vectorField *field_, linearField *Jac, const double epsDist_,
		 const double epsStepSize_, const size_t maxIter_,
		 const bool verbose_=false)
    : fixedPointTrack(field_, Jac, epsDist_, epsStepSize_, maxIter_,
		      verbose) {
    current = gsl_vector_alloc(dim);
    S = gsl_matrix_alloc(dim, dim);
    stepCorr = gsl_vector_calloc(dim);
    targetCorr = gsl_vector_alloc(dim);
    // Set initial step to 1. to avoid singular matrix in correction
    stepPred = gsl_vector_calloc(dim);
    gsl_vector_set(stepPred, dim - 1, 1);
    targetPred = gsl_vector_calloc(dim);
    gsl_vector_set(targetPred, dim - 1, 1.);
  }

  ~fixedPointCont() { gsl_vector_free(stepPred); gsl_vector_free(targetPred); }

  /** \brief Prediction Newton-Raphson step. */
  void predict();

  /** \brief Update state after prediction */
  void applyPredict(const double contStep);

  /** \brief Correction after prediction. */
  void correct();

  /** \brief Correction of initial state. */
  void correct(const gsl_vector *init);

  /** \brief Update correction target vector for fixed point tracking. */
  void updateTargetCorr();
  
  /** \brief Perform one step (correction + prediction) of peudo-arc. continuation. */
  void continueStep(const double contStep);
    
  /** \brief Perform one step (correction + prediction) of peudo-arc. continuation. */
  void continueStep(const double contStep, const gsl_vector *init);
    
  /** \brief Get matrix of the linear system to be solved for correction. */
  void getSystemCorr();

  /** \brief Get matrix of the linear system to be solved for prediction. */
  void getSystemPred();
};


class periodicOrbitTrack : public solutionCorrection {

protected:
  const double intStepCorr; //!< Int. time stepCorr (to be adapted to the period).
  size_t nt;            //!< Total number of integration step.
  gsl_vector_uint *ntShoot; //!< Number of integration step per shoot.
  double dt;            //!< Adapted time step.
  fundamentalMatrixModel * const linMod; //!< Full and linearized model
  const size_t numShoot; //!< Number of shoots to correct/predict orbit

  /** \brief Adapt time step and number of time steps to shooting strategy. */
  virtual void adaptTimeToPeriod();

  /** \brief Adapt time step and number of time steps for a give period. */
  virtual void adaptTimeToPeriod(const double T);
  
public:
  
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitTrack(fundamentalMatrixModel *linMod_, const double epsDist_,
		     const double epsStepCorrSize_, const size_t maxIter_,
		     const double intStepCorr_, const size_t numShoot_=1,
		     const bool verbose_=false)
    : solutionCorrection(epsDist_, epsStepCorrSize_, maxIter_, verbose_),
      linMod(linMod_), intStepCorr(intStepCorr_), numShoot(numShoot_) {
    dim = linMod->getDim();
    ntShoot = gsl_vector_uint_alloc(numShoot); }

  ~periodicOrbitTrack() {
    gsl_vector_uint_free(ntShoot);
    gsl_vector_free(current);
    gsl_vector_free(stepCorr);
    gsl_vector_free(targetCorr);
    gsl_matrix_free(S);
  }

  /** \brief Get number of shoots. */
  size_t getNumShoot() { return numShoot; }

  /** \brief Get current state of tracking. */
  virtual void getCurrentState(gsl_vector *current_);

  /** \brief Set current state of model and fundamental matrix from the current one. */
  virtual void setCurrentState();

  /** \brief Set current state of the problem together with that of the model. */
  virtual void setCurrentState(const gsl_vector *init);

  /** \brief Get the fundamental matrix of the solution. */
  void getStabilityMatrix(gsl_matrix *matrix);

  /** \brief Update targetCorr. */
  virtual void updateTargetCorr() = 0;

  /** \brief Get matrix of the linear system to be solved for correction. */
  virtual void getSystemCorr() = 0;
};


class periodicOrbitCorr : public periodicOrbitTrack {

public:
  
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitCorr(fundamentalMatrixModel *linMod_, const double epsDist_,
		    const double epsStepCorrSize_, const size_t maxIter_,
		    const double intStepCorr_, const size_t numShoot_=1,
		    const bool verbose_=false)
    : periodicOrbitTrack(linMod_, epsDist_, epsStepCorrSize_, maxIter_,
			 intStepCorr_, numShoot_, verbose_) {
    current = gsl_vector_alloc(dim * numShoot + 1);
    stepCorr = gsl_vector_calloc(dim * numShoot + 1);
    targetCorr = gsl_vector_alloc(dim * numShoot + 1);
    S = gsl_matrix_alloc(dim * numShoot + 1, dim * numShoot + 1);
  }

  ~periodicOrbitCorr() { }

  /** \brief Update targetCorr. */
  void updateTargetCorr();

  /** \brief Find fixed point using Newton-Raphson method. */
  void findSolution(const gsl_vector *init);

  /** \brief Get matrix of the linear system to be solved for correction. */
  void getSystemCorr();
};


class periodicOrbitCont : public periodicOrbitTrack {

private:
  gsl_vector *stepPred;         //!< Step of prediction.
  gsl_vector *targetPred;       //!< (0, ..., 0, 1) vector for prediction
  gsl_vector *work;             //!< Workspace vector

  /** \brief Adapt time step and number of time steps to shooting strategy. */
  void adaptTimeToPeriod();
  
  /** \brief Adapt time step and number of time steps for a give period. */
  void adaptTimeToPeriod(const double T);
  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitCont(fundamentalMatrixModel *linMod_, const double epsDist_,
		    const double epsStepCorrSize_, const size_t maxIter_,
		    const double intStepCorr_, const size_t numShoot_=1,
		    const bool verbose_=false)
    : periodicOrbitTrack(linMod_, epsDist_, epsStepCorrSize_, maxIter_,
			 intStepCorr_, numShoot_, verbose_) {
    current = gsl_vector_alloc((dim-1) * numShoot + 2);
    stepCorr = gsl_vector_calloc((dim-1) * numShoot + 2);
    targetCorr = gsl_vector_alloc((dim-1) * numShoot + 2);
    S = gsl_matrix_alloc((dim-1) * numShoot + 2, (dim-1) * numShoot + 2);
    stepPred = gsl_vector_calloc((dim-1) * numShoot + 2);
    // Set initial param step to 1. to avoid singular matrix in correction
    gsl_vector_set(stepPred, (dim-1) * numShoot, 1.);
    targetPred = gsl_vector_calloc((dim-1) * numShoot + 2);
    gsl_vector_set(targetPred, (dim-1) * numShoot, 1.);  // (0,...,0, 1, 0)
    work = gsl_vector_alloc(dim);
  }

  ~periodicOrbitCont() {
    gsl_vector_free(stepPred);
    gsl_vector_free(targetPred);
    gsl_vector_free(work); }

  /** \brief Get extended state vector x(s), lambda. */
  void getExtendedState(gsl_vector *state, const size_t s);

  /** \brief Get current state of tracking. */
  void getCurrentState(gsl_vector *current_);

  /** \brief Set current state of model and fundamental matrix from the current one. */
  void setCurrentState(const size_t s);

  /** \brief Set current state of model and fundamental matrix from the current one. */
  void setCurrentState();

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

  /** \brief Prediction Newton-Raphson step. */
  void predict();

  /** \brief Update state after prediction */
  void applyPredict(const double contStep);

  /** \brief Correction after prediction. */
  void correct();

  /** \brief Correction of initial state. */
  void correct(const gsl_vector *init);

  /** \brief Update correction target vector for periodic orbit tracking. */
  void updateTargetCorr();
  
  /** \brief Perform one step (correction + prediction) of peudo-arc. continuation. */
  void continueStep(const double contStep);
    
  /** \brief Perform one step (correction + prediction) of peudo-arc. continuation. */
  void continueStep(const double contStep, const gsl_vector *init);
    
  /** \brief Get matrix of the linear system to be solved for correction. */
  void getSystemCorr();

  /** \brief Get matrix of the linear system to be solved for prediction. */
  void getSystemPred();

};


/**
p * @}
 */

#endif
