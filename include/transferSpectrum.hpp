#ifndef TRANSFERSPECTRUM_HPP
#define TRANSFERSPECTRUM_HPP

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_sort_vector.h>
#include <arpack++/arsnsym.h>
#include <transferOperator.hpp>
#include <configAR.hpp>


/** \addtogroup transfer
 * @{
 */

/** \file transferSpectrum.hpp
 *  \brief Get spectrum of transferOperator using ARPACK++.
 *   
 *  Analyse the spectrum of the forward and backward transition matrices
 *  of a transferOperator object using ARPACK++.
 */


/*
 * Class declarations
 */


/** \brief Interface from GSL sparse matrices to ARPACK++.
 *
 * Interface from GSL sparse matrices to ARPACK++.
 */
class gsl_spmatrix2AR {
public:
  const gsl_spmatrix *M; //!< GSL sparse matrix


  /** \brief Default constructor. */
  gsl_spmatrix2AR() : M(NULL) {}

  /** \brief Constructor from GSL sparse matrix */
  gsl_spmatrix2AR(const gsl_spmatrix *M_) : M(M_) {}

  /** \brief Matrix-vector product required by ARPACK */
  virtual void MultMv(double *v, double *w);
};


/** \brief Transfer operator spectrum.
 *
 *  Class used to calculate the spectrum of forward and backward
 *  transition matrices of a transferOperator object.
 */
class transferSpectrum {
  const size_t N;  //!< Size of the reduced grid (after mask) != transferOp->N
  const transferOperator *transferOp; //!< Transfer operator of the eigen problem
  int nev;                            //!< Number of eigenvalues and vectors to search;
  configAR config;                    //!< Initial configuration
  /** If true, the problem is stationary and it is no use calculating
   *  the backward transition matrix and final distribution. */
  const bool stationary;
  /** Whether eigenvalues and vectors have been sorted
   *  by descending largest magnitude */
  bool sorted;                        

  
public:
  gsl_vector_complex *EigValForward;  //!< Eigenvalues of forward transition matrix
  gsl_vector_complex *EigValBackward; //!< Eigenvalues of backward transition matrix
  gsl_matrix_complex *EigVecForward;  //!< Eigenvectors of forward transition matrix
  gsl_matrix_complex *EigVecBackward; //!< Eigenvectors of backward transition matrix
  
  
  ARNonSymStdEig<double, gsl_spmatrix2AR > EigProbForward; //!< Eigen value problem for forward transition matrix
  ARNonSymStdEig<double, gsl_spmatrix2AR > EigProbBackward; //!< Eigen value problem for backward transition matrix


  /** \brief Constructor allocating for nev_ eigenvalues and vectors. */
  transferSpectrum(const int nev_, const transferOperator *transferOp_,
		   const configAR cfgAR);
  
  /** \brief Destructor desallocating. */
  ~transferSpectrum();


  /** \brief Get spectrum of forward transition matrices. */
  void getSpectrumForward();
  /** \brief Get spectrum of backward transition matrices. */
  void getSpectrumBackward();
  /** \brief Get spectrum of both forward and backward transition matrices. */
  void getSpectrum();

  /** \brief Get condition number associated with each eigenvalues. */
  gsl_vector *getConditionNumbers();

  /** \brief Sort by largest magnitude. */
  void sort();

  /** \brief Make set of forward and backward eigenvectors biorthonormal. */
  void makeBiorthonormal();

  /** \brief Get weights for two observables. */
  gsl_vector_complex *getWeights(const gsl_vector *f, const gsl_vector *g);

  /** \brief Write forward eigenvalues and eigenvectors. */
  void writeSpectrumForward(const char *EigValForwardFile,
			    const char *EigVecForwardFile,
			    const char *fileFormat) const;
  /** \brief Write backward eigenvalues and eigenvectors. */
  void writeSpectrumBackward(const char *EigValBackwardFile,
			     const char *EigVecBackwardFile,
			     const char *fileFormat) const;
  /** \brief Write forward and backward eigenvalues and eigenvectors. */
  void writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
		     const char *EigValBackwardFile, const char *EigVecBackwardFile,
		     const char *fileFormat) const;

  /** \brief Get number of grid boxes. */
  size_t getN() const { return N; }

  /** \brief Get whether stationary. */
  bool isStationary() const { return stationary; }
  
  /** \brief Get whether forward and backward eigenvalues and vectors are sorted. */
  bool isSorted() const { return sorted; }
  
  /** \brief Get number of eigen values and vectors to look for. */
  int getNev() const { return nev; }

  /** \brief Get configuration. */
  configAR getConfig() const { return  config; }

  /** \brief Set configuration. */
  void setConfig(const configAR newConfig) { config = newConfig; return; }

};


/*
 *  Functions declarations
 */

/** \brief Get spectrum of a nonsymmetric matrix using ARPACK++. */
void getSpectrumAR(int *nev, const size_t N, ARNonSymStdEig<double, gsl_spmatrix2AR > *EigProb,
		   gsl_spmatrix2AR *gsl2AR, configAR cfgAR,
		   gsl_vector_complex **EigVal, gsl_matrix_complex **EigVec);

/** \brief Write spectrum of a nonsymmetric matrix using ARPACK++. */
void writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		     const gsl_vector_complex *EigVal,
		     const gsl_matrix_complex *EigVec,
		     const char *fileFormat);


/**
 * @}
 */

#endif
