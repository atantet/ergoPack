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

class solutionTrack {

protected:
  const size_t dim;         //!< Dimension of the model
  const double epsDist;     //!< Tolerance in the distance to zero
  const double epsStepSize; //!< Tolerance in the size of step
  const size_t maxIter;     //!< Maximum number of iterations
  double errDist;           //!< Distance to zero
  double errStepSize;       //!< Size of step
  gsl_vector *step;         //!< Step correcting the solution
  gsl_vector *target;       //!< Target vector
  size_t numIter;           //!< Number of iterations
  gsl_vector *current;      //!< Current state of tracking
  bool converged;           //!< Flag for convergence of the tracking.

  
public:
  model *mod;                 //!< Model
  stateLinearField *Jacobian; //!< Jacobianobian Vector field

  linearizedModel *linMod; //!< Linearized model
  
  /** \brief Constructor assigning a linearized model and parameters. */
  solutionTrack(linearizedModel *linMod_, const double epsDist_,
	   const double epsStepSize_, const size_t maxIter_)
    : linMod(linMod_), epsDist(epsDist_), epsStepSize(epsStepSize_), maxIter(maxIter_),
      errDist(1.e27), errStepSize(1.e27), numIter(0), dim(linMod_->getDim()),
      converged(false) {}

  /** \brief Return the number of iterations used for tracking. */
  size_t getNumIter() { return numIter; }

  /** \brief Return the last distance to the target. */
  double getDist() { return errDist; }

  /** \brief Return the size of the last step. */
  double getStepSize() { return errStepSize; }

  /** \brief Return the flag for the convergence of the tracking. */
  bool hasConverged() { return converged; }
  
  /** \brief Get current state of tracking. */
  void getCurrentState(gsl_vector *current_);

  /** \brief Correct with the step obtained from the Newton step. */
  void applyStep(const double damping=1.);

  /** \brief Destructor. */
  virtual ~solutionTrack(){};

  /** \brief Set current state of the problem together with that of the model. */
  virtual void setCurrentState(const gsl_vector *init) = 0;

  /** \brief Update target. */
  virtual void updateTarget() = 0;

  /** \brief Perform one Newton-Raphson step. */
  virtual void NewtonStep() = 0;

  /** \brief Find fixed point using Newton-Raphson method. */
  virtual void findSolution(const gsl_vector *init) = 0;

  /** \brief Get stability matrix associated with the solution. */
  virtual void getStabilityMatrix(gsl_matrix *matrix) = 0;

};



class fixedPointTrack : public solutionTrack {

  gsl_vector *init;
  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  fixedPointTrack(linearizedModel *linMod_, const double epsDist_,
		  const double epsStepSize_, const size_t maxIter_)
    : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_) {
    current = gsl_vector_alloc(dim);
    step = gsl_vector_alloc(dim);
    target = gsl_vector_alloc(dim);}

  ~fixedPointTrack()
  { gsl_vector_free(current); gsl_vector_free(step); gsl_vector_free(target); }

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

  /** \brief Perform one Newton-Raphson step. */
  void NewtonStep();

  /** \brief Update target. */
  void updateTarget();

  /** \brief Find fixed point using Newton-Raphson method. */
  void findSolution(const gsl_vector *init);

  /** \brief Get the Jacobian of the solution. */
  void getStabilityMatrix(gsl_matrix *matrix);

};


class periodicOrbitTrack : public solutionTrack {

private:
  const double intStep; //!< Integration time step (to be adapted to the period).
  size_t nt;            //!< Number of integration steps.
  double dt;            //!< Adapted time step.
  gsl_matrix *S;        //!< Matrix of the linear system to solve.
  gsl_vector_view currentState; //!< View on the state part of the current state.
  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  periodicOrbitTrack(linearizedModel *linMod_, const double epsDist_,
		     const double epsStepSize_, const size_t maxIter_,
		     const double intStep_)
    : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_), intStep(intStep_) {
    current = gsl_vector_alloc(dim + 1);
    currentState = gsl_vector_subvector(current, 0, dim);
    step = gsl_vector_alloc(dim + 1);
    target = gsl_vector_alloc(dim + 1);
    S = gsl_matrix_alloc(dim + 1, dim + 1); }

  ~periodicOrbitTrack() {
    gsl_vector_free(current);
    gsl_vector_free(step);
    gsl_vector_free(target);
    gsl_matrix_free(S); }

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

  /** \brief Perform one Newton-Raphson step. */
  void NewtonStep();

  /** \brief Update target. */
  void updateTarget();

  /** \brief Find fixed point using Newton-Raphson method. */
  void findSolution(const gsl_vector *init);

  /** \brief Get the fundamental matrix of the solution. */
  void getStabilityMatrix(gsl_matrix *matrix);

  /** \brief Get matrix of the linear system to be solved. */
  void getLinearSystem();
};


class fixedPointCont : public solutionTrack {

private:
  gsl_matrix *S;        //!< Matrix of the linear system to solve.
  gsl_vector_view currentState; //!< View on the state part of the current state.
  
public:
  /** \brief Constructor assigning a linearized model and parameters. */
  fixedPointCont(linearizedModel *linMod_, const double epsDist_,
		 const double epsStepSize_, const size_t maxIter_)
    : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_) {
    current = gsl_vector_alloc(dim + 1);
    currentState = gsl_vector_subvector(current, 0, dim);
    step = gsl_vector_alloc(dim + 1);
    target = gsl_vector_alloc(dim + 1);
    S = gsl_matrix_alloc(dim + 1, dim + 1); }

  ~fixedPointCont() {
    gsl_vector_free(current);
    gsl_vector_free(step);
    gsl_vector_free(target);
    gsl_matrix_free(S); }

  /** \brief Set current state of the problem together with that of the model. */
  void setCurrentState(const gsl_vector *init);

  /** \brief Prediction Newton-Raphson step. */
  void predict();

  /** \brief Correction by continuation. */
  void correct();

  /** \brief Update target. */
  void updateTarget();

  /** \brief Find fixed point using Newton-Raphson method. */
  void findSolution(const gsl_vector *init);

  /** \brief Get the fundamental matrix of the solution. */
  void getStabilityMatrix(gsl_matrix *matrix);

  /** \brief Get matrix of the linear system to be solved. */
  void getLinearSystem();
};


// class solutionTrack {

// protected:
//   const size_t dim;         //!< Dimension of the model
//   const double epsDist;     //!< Tolerance in the distance to zero
//   const double epsStepSize; //!< Tolerance in the size of step
//   const size_t maxIter;     //!< Maximum number of iterations
//   double errDist;           //!< Distance to zero
//   double errStepSize;       //!< Size of step
//   gsl_vector *step;         //!< Step correcting the solution
//   gsl_vector *target;       //!< Target vector
//   size_t numIter;           //!< Number of iterations
//   gsl_vector *current;      //!< Current state of tracking
//   bool converged;           //!< Flag for convergence of the tracking.

  
// public:
//   linearizedModel *linMod; //!< Linearized model
  
//   /** \brief Constructor assigning a linearized model and parameters. */
//   solutionTrack(linearizedModel *linMod_, const double epsDist_,
// 	   const double epsStepSize_, const size_t maxIter_)
//     : linMod(linMod_), epsDist(epsDist_), epsStepSize(epsStepSize_), maxIter(maxIter_),
//       errDist(1.e27), errStepSize(1.e27), numIter(0), dim(linMod_->getDim()),
//       converged(false) {}

//   /** \brief Return the number of iterations used for tracking. */
//   size_t getNumIter() { return numIter; }

//   /** \brief Return the last distance to the target. */
//   double getDist() { return errDist; }

//   /** \brief Return the size of the last step. */
//   double getStepSize() { return errStepSize; }

//   /** \brief Return the flag for the convergence of the tracking. */
//   bool hasConverged() { return converged; }
  
//   /** \brief Get current state of tracking. */
//   void getCurrentState(gsl_vector *current_);

//   /** \brief Correct with the step obtained from the Newton step. */
//   void applyStep(const double damping=1.);

//   /** \brief Destructor. */
//   virtual ~solutionTrack(){};

//   /** \brief Set current state of the problem together with that of the model. */
//   virtual void setCurrentState(const gsl_vector *init) = 0;

//   /** \brief Update target. */
//   virtual void updateTarget() = 0;

//   /** \brief Perform one Newton-Raphson step. */
//   virtual void NewtonStep() = 0;

//   /** \brief Find fixed point using Newton-Raphson method. */
//   virtual void findSolution(const gsl_vector *init) = 0;

//   /** \brief Get stability matrix associated with the solution. */
//   virtual void getStabilityMatrix(gsl_matrix *matrix) = 0;

// };



// class fixedPointTrack : public solutionTrack {

//   gsl_vector *init;
  
// public:
//   /** \brief Constructor assigning a linearized model and parameters. */
//   fixedPointTrack(linearizedModel *linMod_, const double epsDist_,
// 		  const double epsStepSize_, const size_t maxIter_)
//     : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_) {
//     current = gsl_vector_alloc(dim);
//     step = gsl_vector_alloc(dim);
//     target = gsl_vector_alloc(dim);}

//   ~fixedPointTrack()
//   { gsl_vector_free(current); gsl_vector_free(step); gsl_vector_free(target); }

//   /** \brief Set current state of the problem together with that of the model. */
//   void setCurrentState(const gsl_vector *init);

//   /** \brief Perform one Newton-Raphson step. */
//   void NewtonStep();

//   /** \brief Update target. */
//   void updateTarget();

//   /** \brief Find fixed point using Newton-Raphson method. */
//   void findSolution(const gsl_vector *init);

//   /** \brief Get the Jacobian of the solution. */
//   void getStabilityMatrix(gsl_matrix *matrix);

// };


// class periodicOrbitTrack : public solutionTrack {

// private:
//   const double intStep; //!< Integration time step (to be adapted to the period).
//   size_t nt;            //!< Number of integration steps.
//   double dt;            //!< Adapted time step.
//   gsl_matrix *S;        //!< Matrix of the linear system to solve.
//   gsl_vector_view currentState; //!< View on the state part of the current state.
  
// public:
//   /** \brief Constructor assigning a linearized model and parameters. */
//   periodicOrbitTrack(linearizedModel *linMod_, const double epsDist_,
// 		     const double epsStepSize_, const size_t maxIter_,
// 		     const double intStep_)
//     : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_), intStep(intStep_) {
//     current = gsl_vector_alloc(dim + 1);
//     currentState = gsl_vector_subvector(current, 0, dim);
//     step = gsl_vector_alloc(dim + 1);
//     target = gsl_vector_alloc(dim + 1);
//     S = gsl_matrix_alloc(dim + 1, dim + 1); }

//   ~periodicOrbitTrack() {
//     gsl_vector_free(current);
//     gsl_vector_free(step);
//     gsl_vector_free(target);
//     gsl_matrix_free(S); }

//   /** \brief Set current state of the problem together with that of the model. */
//   void setCurrentState(const gsl_vector *init);

//   /** \brief Perform one Newton-Raphson step. */
//   void NewtonStep();

//   /** \brief Update target. */
//   void updateTarget();

//   /** \brief Find fixed point using Newton-Raphson method. */
//   void findSolution(const gsl_vector *init);

//   /** \brief Get the fundamental matrix of the solution. */
//   void getStabilityMatrix(gsl_matrix *matrix);

//   /** \brief Get matrix of the linear system to be solved. */
//   void getLinearSystem();
// };


// class fixedPointCont : public solutionTrack {

// private:
//   gsl_matrix *S;        //!< Matrix of the linear system to solve.
//   gsl_vector_view currentState; //!< View on the state part of the current state.
  
// public:
//   /** \brief Constructor assigning a linearized model and parameters. */
//   fixedPointCont(linearizedModel *linMod_, const double epsDist_,
// 		 const double epsStepSize_, const size_t maxIter_)
//     : solutionTrack(linMod_, epsDist_, epsStepSize_, maxIter_) {
//     current = gsl_vector_alloc(dim + 1);
//     currentState = gsl_vector_subvector(current, 0, dim);
//     step = gsl_vector_alloc(dim + 1);
//     target = gsl_vector_alloc(dim + 1);
//     S = gsl_matrix_alloc(dim + 1, dim + 1); }

//   ~fixedPointCont() {
//     gsl_vector_free(current);
//     gsl_vector_free(step);
//     gsl_vector_free(target);
//     gsl_matrix_free(S); }

//   /** \brief Set current state of the problem together with that of the model. */
//   void setCurrentState(const gsl_vector *init);

//   /** \brief Prediction Newton-Raphson step. */
//   void predict();

//   /** \brief Correction by continuation. */
//   void correct();

//   /** \brief Update target. */
//   void updateTarget();

//   /** \brief Find fixed point using Newton-Raphson method. */
//   void findSolution(const gsl_vector *init);

//   /** \brief Get the fundamental matrix of the solution. */
//   void getStabilityMatrix(gsl_matrix *matrix);

//   /** \brief Get matrix of the linear system to be solved. */
//   void getLinearSystem();
// };


/**
 * @}
 */

#endif
