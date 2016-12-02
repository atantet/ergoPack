#ifndef ODECONT_HPP
#define ODECONT_HPP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

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

  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  solutionCorrection(const double epsDist_, const double epsStepCorrSize_,
		const size_t maxIter_)
    : epsDist(epsDist_), epsStepCorrSize(epsStepCorrSize_), maxIter(maxIter_),
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
  
  /** \brief Get current state of tracking. */
  void getCurrentState(gsl_vector *current_);

  /** \brief Perform a Newton step to find a fixed point. */
  void NewtonStep();
  
  /** \brief Correct with the stepCorr obtained from the Newton stepCorr. */
  void applyCorr(const double damping=1.);

  /** \brief Destructor. */
  virtual ~solutionCorrection(){};

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
		  const double epsStepCorrSize_, const size_t maxIter_)
    : solutionCorrection(epsDist_, epsStepCorrSize_, maxIter_), field(field_),
      Jacobian(Jac) { dim = Jac->getRows(); }

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
		  const double epsStepCorrSize_, const size_t maxIter_)
    : fixedPointTrack(field_, Jac, epsDist_, epsStepCorrSize_, maxIter_) {
    current = gsl_vector_alloc(dim);
    S = gsl_matrix_alloc(dim, dim);
    stepCorr = gsl_vector_alloc(dim);
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
		 const double epsStepSize_, const size_t maxIter_)
    : fixedPointTrack(field_, Jac, epsDist_, epsStepSize_, maxIter_) {
    current = gsl_vector_alloc(dim);
    S = gsl_matrix_alloc(dim, dim);
    stepCorr = gsl_vector_alloc(dim);
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


class periodicOrbitTrack : public solutionCorrection {

protected:
  const double intStepCorr; //!< Int. time stepCorr (to be adapted to the period).
  size_t nt;            //!< Number of integration stepCorrs.
  double dt;            //!< Adapted time stepCorr.
  fundamentalMatrixModel * const linMod; //!< Full and linearized model
  
public:
  
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitTrack(fundamentalMatrixModel *linMod_, const double epsDist_,
		     const double epsStepCorrSize_, const size_t maxIter_,
		     const double intStepCorr_)
    : solutionCorrection(epsDist_, epsStepCorrSize_, maxIter_), linMod(linMod_),
      intStepCorr(intStepCorr_) { dim = linMod->getDim(); }

  ~periodicOrbitTrack() {
    gsl_vector_free(current);
    gsl_vector_free(stepCorr);
    gsl_vector_free(targetCorr);
    gsl_matrix_free(S);
  }

  /** \brief Set current state of model and fundamental matrix from the current one. */
  void setCurrentState();

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

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
		     const double intStepCorr_)
    : periodicOrbitTrack(linMod_, epsDist_, epsStepCorrSize_, maxIter_,
			 intStepCorr_) {
    current = gsl_vector_alloc(dim + 1);
    stepCorr = gsl_vector_alloc(dim + 1);
    targetCorr = gsl_vector_alloc(dim + 1);
    S = gsl_matrix_alloc(dim + 1, dim + 1);
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

public:
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitCont(fundamentalMatrixModel *linMod_, const double epsDist_,
		    const double epsStepCorrSize_, const size_t maxIter_,
		    const double intStepCorr_)
    : periodicOrbitTrack(linMod_, epsDist_, epsStepCorrSize_, maxIter_,
			 intStepCorr_) {
    current = gsl_vector_alloc(dim + 1);
    stepCorr = gsl_vector_alloc(dim + 1);
    targetCorr = gsl_vector_alloc(dim + 1);
    S = gsl_matrix_alloc(dim + 1, dim + 1);
    stepPred = gsl_vector_calloc(dim + 1);
    // Set initial param step to 1. to avoid singular matrix in correction
    gsl_vector_set(stepPred, dim-1, 1);
    targetPred = gsl_vector_calloc(dim + 1);
    gsl_vector_set(targetPred, dim-1, 1.);  // (0,...,0, 1, 0)
  }

  ~periodicOrbitCont() { gsl_vector_free(stepPred); gsl_vector_free(targetPred); }

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
 * @}
 */

#endif
