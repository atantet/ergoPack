#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

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
#include <ergoPack/transferOperator.hpp>

/** \file transferSpectrum.hpp
 *  \brief Get spectrum of transferOperator using ARPACK++.
 *   
 *  Analyse the spectrum of the forward and backward transition matrices
 *  of a transferOperator object using ARPACK++.
 */


/*
 * Class declarations
 */


/**
 * \brief Utility structure used to give configuration options to ARPACK++.
 * 
 * Utility structure used to give configuration options to ARPACK++.
 */
typedef struct {
  char *which; //!< Which eigenvalues to look for. 'LM' for Largest Magnitude
  int ncv;           //!< The number of Arnoldi vectors generated at each iteration of ARPACK
  double tol;        //!< The relative accuracy to which eigenvalues are to be determined
  int maxit;         //!< The maximum number of iterations allowed
  double *resid;     //!< A starting vector for the Arnoldi process
  bool AutoShift;    /**< Use ARPACK++ generated exact shifts for the implicit restarting
		      *   of the Arnoldi or one supplied by user */
} configAR;
/** Declare default structure looking for largest magnitude eigenvalues */
configAR defaultCfgAR = {"LM", 0, 0., 0, NULL, true};


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
  const size_t N;  //!< Size of the grid
  const transferOperator *transferOp; //!< Transfer operator of the eigen problem
  const int nev;                      //!< Number of eigenvalues and vectors to search;
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
  
  /** Eigen value problem for forward transition matrix */
  ARNonSymStdEig<double, gsl_spmatrix2AR > EigProbForward;
  /** Eigen value problem for backward transition matrix */
  ARNonSymStdEig<double, gsl_spmatrix2AR > EigProbBackward; 

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
			    const char *EigVecForwardFile) const;
  /** \brief Write backward eigenvalues and eigenvectors. */
  void writeSpectrumBackward(const char *EigValBackwardFile,
			     const char *EigVecBackwardFile) const;
  /** \brief Write forward and backward eigenvalues and eigenvectors. */
  void writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
		     const char *EigValBackwardFile, const char *EigVecBackwardFile) const;

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
void getSpectrumAR(const int nev, ARNonSymStdEig<double, gsl_spmatrix2AR > *EigProb,
		   gsl_spmatrix2AR *gsl2AR, configAR cfgAR,
		   gsl_vector_complex *EigVal, gsl_matrix_complex *EigVec);

/** \brief Write spectrum of a nonsymmetric matrix using ARPACK++. */
void writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		     const gsl_vector_complex *EigVal,
		     const gsl_matrix_complex *EigVec);


/*
 * Constructors and destructors definitions
 */

/**
 * Constructor allocating space for a given number of eigenvalues and vectors
 * for a given transferOperator.
 * \param[in] nev_        Number of eigenvalues and eigenvectors for which to allocate.
 * \param[in] transferOp_ Pointer to the transferOperator on which to solve the eigen problem.
 * \param[in] cfgAR       Configuration data used by ARPACK++ for the eigen problem.
 */
transferSpectrum::transferSpectrum(const int nev_, const transferOperator *transferOp_,
				   const configAR cfgAR=defaultCfgAR)
  : N(transferOp_->getN()), nev(nev_), transferOp(transferOp_), config(cfgAR),
    stationary(transferOp_->isStationary()), sorted(false),
    EigValForward(NULL), EigVecForward(NULL), EigValBackward(NULL), EigVecBackward(NULL) {}


/**
 * Destructor desallocates. 
 */
transferSpectrum::~transferSpectrum()
{
  //! Free memory for forward eigenvalues and eigenvectors
  if (EigValForward)
    gsl_vector_complex_free(EigValForward);
  if (EigVecForward)
    gsl_matrix_complex_free(EigVecForward);
  
  //! Free memory for backward eigenvalues and eigenvectors
  if (EigValBackward)
    gsl_vector_complex_free(EigValBackward);
  if (EigVecBackward)
    gsl_matrix_complex_free(EigVecBackward);
}


/*
 * Methods definitions
 */

/**
 * Get spectrum of forward transition matrix.
 */
void
transferSpectrum::getSpectrumForward()
{
  gsl_spmatrix *cpy;
  gsl_spmatrix2AR gsl2AR;

  // Allocate
  EigValForward = gsl_vector_complex_alloc(nev);
  EigVecForward = gsl_matrix_complex_alloc(N, nev);

  /** Get transpose of forward transition matrix in CCS 
   *  Transposing is trivial since the transition matrix is in CRS format.
   *  However, it is more secure to avoid directly changing the type of
   *  the transition matrix. */
  cpy = gsl_spmatrix_alloc_nzmax(transferOp->P->size2, transferOp->P->size1, 0, GSL_SPMATRIX_CCS);
  cpy->innerSize = transferOp->P->innerSize;
  cpy->outerSize = transferOp->P->outerSize;
  cpy->i = transferOp->P->i;
  cpy->data = transferOp->P->data;
  cpy->p = transferOp->P->p;
  cpy->nzmax = transferOp->P->nzmax;
  cpy->nz = transferOp->P->nz;
  gsl2AR = gsl_spmatrix2AR(cpy);

  //! Solve eigen value problem using user-defined ARPACK++
  getSpectrumAR(nev, &EigProbForward, &gsl2AR, config, EigValForward, EigVecForward);

  return;
}


/**
 * Get spectrum of backward transition matrix.
 */
void
transferSpectrum::getSpectrumBackward()
{
  gsl_spmatrix *cpy;
  gsl_spmatrix2AR gsl2AR;
  gsl_vector_complex_view view;
  gsl_complex element;

  // Allocate
  // Allocate
  EigValBackward = gsl_vector_complex_alloc(nev);
  EigVecBackward = gsl_matrix_complex_alloc(N, nev);

  if (!stationary)
    {
      /** Get transpose of backward transition matrix in ARPACK CCS format */
      cpy = gsl_spmatrix_alloc_nzmax(transferOp->Q->size2, transferOp->Q->size1,
				     0, GSL_SPMATRIX_CCS);
      cpy->innerSize = transferOp->Q->innerSize;
      cpy->outerSize = transferOp->Q->outerSize;
      cpy->i = transferOp->Q->i;
      cpy->data = transferOp->Q->data;
      cpy->p = transferOp->Q->p;
      cpy->nzmax = transferOp->Q->nzmax;
      cpy->nz = transferOp->Q->nz;
      gsl2AR = gsl_spmatrix2AR(cpy);
    }
  else
    {
      /** In the stationary case, the adjoint eigenvectors can be calculated
       *  as left eigenvectors of the forward transition matrix
       *  divided by the stationary distribution */
      gsl2AR = gsl_spmatrix2AR(transferOp->P);
    }
  
  /** Get eigenvalues and vectors of backward transition matrix */
  getSpectrumAR(nev, &EigProbBackward, &gsl2AR, config, EigValBackward, EigVecBackward);

  if (stationary)
    {
      /** Divide eigenvectors (real and imaginary parts) by stationary distribution */
      for (size_t i = 0; i < N; i++)
	{
	  if (gsl_vector_get(transferOp->rho0, i) > 0)
	    {
	      element = gsl_complex_rect(1. / gsl_vector_get(transferOp->rho0, i), 0.);
	      view = gsl_matrix_complex_row(EigVecBackward, i);
	      gsl_vector_complex_scale(&view.vector, element);
  	    }
  	}
    }

  return;
}


/**
 * Get spectrum of transfer operator,
 * including the complex eigenvalues,
 * the left eigenvectors of the forward transition matrix
 * and the right eigenvectors of the backward transitioin matrix.
 * The vectors of eigenvalues and eigenvectors should not be preallocated.
 */
void
transferSpectrum::getSpectrum()
{
  /** Get spectrum of forward transition matrix */
  getSpectrumForward();
  
  /** Get spectrum of backward transition matrix */
  getSpectrumBackward();
  
  return;
}
    

/**
 * Get condition numbers associated with each eigenvalue.
 * The condition number associated with eigenvalue \f$i\f$ is
 * \f$\kappa_i = \frac{\|\psi_i\|_{\rho_s} \|\psi^*_i\|_{\rho_f}}
 * {\left<\psi_i, \psi_i^*\right>_{\rho_t}}\f$,
 * where \f$\psi_i\f$ and \f$\psi_i^*\f$ are the forward and backward (adjoint) eigenvectors,
 * respectively and norm is for \f$L^2\f$ w.r.t the initial or final distributions.
 * The condition number is equal to the inverse of the cosine of the angle
 * between the two vectors and gives a measure of the nonnormality associated
 * with a particular pair of eigenvectors.
 * \return Condition numbers associated with each pair of eigenvectors.
 */
gsl_vector *
transferSpectrum::getConditionNumbers()
{
  gsl_vector *conditionNumbers = gsl_vector_alloc(nev);
  double normForward, normBackward, inner;

  //! Get norms of eigenvectors
  for (size_t ev = 0; ev < (size_t) nev; ev++)
    {
      //! Get norm of forward eigenvector
      gsl_vector_complex_const_view viewFor
	= gsl_matrix_complex_const_column(EigVecForward, ev);
      normForward = gsl_vector_complex_get_norm(&viewFor.vector,
						transferOp->rho0);

      //! Get norm of backward eigenvector.
      gsl_vector_complex_const_view viewBack
	= gsl_matrix_complex_const_column(EigVecBackward, ev);
      normBackward = gsl_vector_complex_get_norm(&viewBack.vector,
						 transferOp->rhof);

      //! Divide by their inner product (in case not made biorthonormal)
      inner = GSL_REAL(gsl_vector_complex_get_inner_product(&viewFor.vector,
							    &viewBack.vector,
							    transferOp->rhof));

      //! Set condition number
      gsl_vector_set(conditionNumbers, ev, normForward * normBackward / inner);
    }

  return conditionNumbers;
}


/**
 * Get weights associated with the projection of two observables
 * on the backward and forward eigenvectors, respectively, i.e.
 * \f$ w_i = \left<f, \psi_k^* \right>_{\rho_0} \left<\psi_k, g \right>_{\rho_f} \f$.
 * \param[in] f Vector representing the first observable, projected on the backward eigenvectors.
 * \param[in] g Vector representing the second observable, projected on the forward eigenvectors.
 * \return Weights associated with each pair of eigenvectors for the two observables.
 */
gsl_vector_complex *
transferSpectrum::getWeights(const gsl_vector *f, const gsl_vector *g)
{
  gsl_vector_complex *weights, *fc, *gc;
  gsl_complex weight;

  if ((f->size != g->size) || (f->size != N))
    {
      std::cerr << "transferSpectrum::getWeights, observables\
must have the same size as the grid." << std::endl;
      throw std::exception();
    }

  // Convert observable vectors to complex
  fc = gsl_vector_complex_calloc(f->size);
  gsl_vector_complex_memcpy_real(fc, f);
  gc = gsl_vector_complex_calloc(g->size);
  gsl_vector_complex_memcpy_real(gc, g);

  // Allocate and get weights
  weights = gsl_vector_complex_alloc(N);
  for (size_t ev = 0; ev < (size_t) nev; ev++)
    {
      // Project on backward eigenvector
      gsl_vector_complex_const_view viewBack
	= gsl_matrix_complex_const_column(EigVecBackward, ev);
      weight = gsl_vector_complex_get_inner_product(fc, &viewBack.vector, transferOp->rho0);

      // Project on forward eigenvector
      gsl_vector_complex_const_view viewFor
	= gsl_matrix_complex_const_column(EigVecForward, ev);
      weight = gsl_complex_mul(weight,
			       gsl_vector_complex_get_inner_product(&viewFor.vector, gc,
								    transferOp->rhof));

      // Set weight
      gsl_vector_complex_set(weights, ev, weight);
    }

  // Free
  gsl_vector_complex_free(fc);
  gsl_vector_complex_free(gc);

  return weights;
}


/**
 * Sort by largest maginitude.
 */
void
transferSpectrum::sort()
{
  gsl_vector_complex *tmpEigValFor = gsl_vector_complex_alloc(nev);
  gsl_matrix_complex *tmpEigVecFor = gsl_matrix_complex_alloc(N, nev);
  gsl_matrix_complex *tmpEigVecBack = gsl_matrix_complex_alloc(N, nev);
  gsl_vector_complex *tmpEigValBack = gsl_vector_complex_alloc(nev);
  gsl_vector *absEigVal;
  gsl_permutation *sort_idx = gsl_permutation_alloc(nev);
  gsl_complex tmp;
  size_t idx;
  
  //! Test that both forward and backward eigen problems have been solved.
  if (!EigValForward  || !EigVecForward)
    {
      std::cerr << "transferSpectrum::sort, forward eigen problem\
must be solved first before to sort" << std::endl;
      throw std::exception();
    }
  else if (!EigValBackward || !EigVecBackward)
    {
      std::cerr << "transferSpectrum::sort, backward eigen problem\
must be solved first before to sort" << std::endl;
      throw std::exception();
    }
  
  //! Sort forward eigenvalues by decreasing absolute value
  absEigVal = gsl_vector_complex_abs(EigValForward);
  gsl_sort_vector_index(sort_idx, absEigVal);
  gsl_permutation_reverse(sort_idx);
  gsl_permute_vector_complex(sort_idx, EigValForward);
  gsl_permute_matrix_complex(sort_idx, EigVecForward, 1);

  /** Sort backward vectors by correspondance to forward counterparts
   *  (since different eigenvalues may have the same magnitude). */
  gsl_matrix_complex_memcpy(tmpEigVecBack, EigVecBackward);
  gsl_vector_complex_memcpy(tmpEigValBack, EigValBackward);
  for (size_t ev = 0; ev < (size_t) nev; ev++)
    {
      //! Get distance from eigenvalue
      gsl_vector_complex_memcpy(tmpEigValFor, EigValForward);
      tmp = gsl_complex_negative(gsl_complex_conjugate(gsl_vector_complex_get(tmpEigValBack, ev)));
      gsl_vector_complex_add_constant(tmpEigValFor, tmp);
      idx = gsl_vector_complex_min_index(tmpEigValFor);

      //! Sort backward eigenvalue and eigenvector
      gsl_vector_complex_set(EigValBackward, idx,
			     gsl_vector_complex_get(tmpEigValBack, ev));
      gsl_vector_complex_const_view view =
	gsl_matrix_complex_const_column(tmpEigVecBack, ev);
      gsl_matrix_complex_set_col(EigVecBackward, idx, &view.vector);
    }

  //! Free
  gsl_vector_complex_free(tmpEigValFor);
  gsl_matrix_complex_free(tmpEigVecFor);
  gsl_vector_complex_free(tmpEigValBack);
  gsl_matrix_complex_free(tmpEigVecBack);
  gsl_vector_free(absEigVal);
  gsl_permutation_free(sort_idx);
  
  //! Mark that sorting has been performed
  sorted = true;
  
  return;
}
 
 
/**
 * Make set of forward and backward eigenvectors biorthonormal.
 */
 void
 transferSpectrum::makeBiorthonormal()
 {
   gsl_complex inner;
      
   /** Test that both forward and backward eigen problems have been solved
    *  and that eigenvalues and eigenvectors have been sorted. */
   if (!EigValForward  || !EigVecForward)
    {
      std::cerr << "transferSpectrum::makeBiorthonormal, forward eigen problem\
must be solved first before to make biorthonormal set" << std::endl;
      throw std::exception();
    }
   else if (!EigValBackward || !EigVecBackward)
     {
       std::cerr << "transferSpectrum::makeBiorthonormal, backward eigen problem\
must be solved first before to make biorthonormal set" << std::endl;
       throw std::exception();
     }
   else if (!sorted)
     {
       sort();
     }
   
   /** Normalize backward eigenvectors by the scalar product
    *  with forward eigenvectors to make biorthonormal set. */
   for (size_t ev = 0; ev < (size_t) nev; ev++)
     {
       //! Get inner product of forward and backward eigenvectors
       gsl_vector_complex_const_view viewFor
   	 = gsl_matrix_complex_const_column(EigVecForward, ev);
       gsl_vector_complex_view viewBack
   	 = gsl_matrix_complex_column(EigVecBackward, ev);
       inner = gsl_vector_complex_get_inner_product(&viewFor.vector,
   						    &viewBack.vector,
   						    transferOp->rhof);
       //! Divide backward eigenvector by conjugate of inner product
       inner = gsl_complex_conjugate(inner);
       inner = gsl_complex_inverse(inner);
       gsl_vector_complex_scale(&viewBack.vector, inner);
     }
   
   return;
}


/**
 * Write eigenvalues and eigenvectors of forward transition matrix.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] EigVecForwardFile  File name of the file to print forward eigenvectors.
 * \return                       Exit status.
 */
void
transferSpectrum::writeSpectrumForward(const char *EigValForwardFile,
				       const char *EigVecForwardFile) const
{
  FILE *streamEigVal, *streamEigVec;
  
  /** Open files for forward */
  if (!(streamEigVal = fopen(EigValForwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream for writing forward eigenvalues");
    }
  if (!(streamEigVec = fopen(EigVecForwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream for writing forward eigenvectors");
    }

  /** Write forward */
  writeSpectrumAR(streamEigVal, streamEigVec, EigValForward, EigVecForward);

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);

  return;
}


/**
 * Write eigenvalues and eigenvectors of backward transition matrix.
 * \param[in] EigValBackwardFile  File name of the file to print backward eigenvalues.
 * \param[in] EigVecBackwardFile  File name of the file to print backward eigenvectors.
 * \return                       Exit status.
 */
void
transferSpectrum::writeSpectrumBackward(const char *EigValBackwardFile,
					const char *EigVecBackwardFile) const
{
  FILE *streamEigVal, *streamEigVec;
  
  /** Open files back backward */
  if (!(streamEigVal = fopen(EigValBackwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream back writing backward eigenvalues");
    }
  if (!(streamEigVec = fopen(EigVecBackwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream back writing backward eigenvectors");
    }

  /** Write backward */
  writeSpectrumAR(streamEigVal, streamEigVec, EigValBackward, EigVecBackward);

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);

  return;
}


/**
 * Write complex eigenvalues and eigenvectors
 * of forward and backward transition matrices.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] EigVecForwardFile  File name of the file to print forward eigenvectors.
 * \param[in] EigValBackwardFile File name of the file to print backward eigenvalues.
 * \param[in] EigVecBackwardFile File name of the file to print backward eigenvectors.
 * \return                       Exit status.
 */
void
transferSpectrum::writeSpectrum(const char *EigValForwardFile,
				const char *EigVecForwardFile,
				const char *EigValBackwardFile,
				const char *EigVecBackwardFile) const
{
  //! Write forward eigenvalues and eigenvectors
  writeSpectrumForward(EigValForwardFile, EigVecForwardFile);

  //! Write backward eigenvalues and eigenvectors
  writeSpectrumBackward(EigValBackwardFile, EigVecBackwardFile);
  
  return;
}


/**
 * Matrix-vector product required by ARPACK++ when using user-defined matrices
 * (GSL sparse matrices). Modified version of gsl_spblas_dgemv in gsl_spmatrix.h .
 * \param[in]  v Vector to multiply.
 * \param[out] w Vector storing the result of multiplication.
 */
void
gsl_spmatrix2AR::MultMv(double *v, double *w)
{
  size_t j, outerIdx, p, n;
  
  /* form y := 0 */
  for (j = 0; j < M->size1; ++j)
    {
      w[j] = 0.0;
    }

  /* form w := M * v */
  if (GSL_SPMATRIX_ISCRS(M))
    {
      /* (row, column) = (outerIdx, M->i) */
      for (outerIdx = 0; outerIdx < M->outerSize; ++outerIdx)
	{
	  for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; ++p)
	    {
	      w[outerIdx] += M->data[p] * v[M->i[p]];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCCS(M))
    {
      /* (row, column) = (M->i, outerIdx) */
      for (outerIdx = 0; outerIdx < M->outerSize; ++outerIdx)
	{
	  for (p = M->p[outerIdx]; p < M->p[outerIdx + 1]; ++p)
	    {
	      w[M->i[p]] += M->data[p] * v[outerIdx];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISTRIPLET(M))
    {
      /* (row, column) = (M->i, M->p) */
      for (n = 0; n < M->nz; ++n)
	{
	  w[M->i[n]] += M->data[n] * v[M->p[n]];
	}
    }
  
  return;
}


/*
 * Functions definitions
 */

/**
 *
 * Get spectrum of a nonsymmetric matrix using ARPACK++.
 * \param[in]     nev        Number of eigenvalues and eigenvectors to find.
 * \param[in,out] EigProb    Eigen problem to use.
 * \param[in]     gsl2AR     Object interfacing the GSL sparse matrix to ARPACK++.
 * \param[in]     cfgAR      Configuration options passed as a configAR object.
 * \param[out]    EigVal     Found eigenvalues.
 * \param[out]    EigVec     Found eigenvectors.
*/
void
getSpectrumAR(const int nev, ARNonSymStdEig<double, gsl_spmatrix2AR > *EigProb,
	      gsl_spmatrix2AR *gsl2AR, configAR cfgAR,
	      gsl_vector_complex *EigVal, gsl_matrix_complex *EigVec)
{
  size_t N = EigVec->size1;
  gsl_complex element;

  // Allocate vectors given to ARPACK++
  double *EigValReal = new double [nev+1];
  double *EigValImag = new double [nev+1];
  double *EigVecRealImag = new double [(nev+2) * N];

  //! Define non-hermitian eigenvalue problem
  EigProb->DefineParameters(gsl2AR->M->size1, nev, gsl2AR,
			    &gsl_spmatrix2AR::MultMv,
			    cfgAR.which, cfgAR.ncv, cfgAR.tol,
			    cfgAR.maxit, cfgAR.resid, cfgAR.AutoShift);

  //! Find eigenvalues and left eigenvectors with ARPACK++
  EigProb->EigenValVectors(EigVecRealImag, EigValReal, EigValImag);

  //! Save eigenvalues and eigenvectors and their complex conjugate
  for (size_t ev = 0; ev < (size_t) nev; ev++)
    {
      element = gsl_complex_rect(EigValReal[ev], EigValImag[ev]);
      gsl_vector_complex_set(EigVal, ev, element);
      
      // Add real part of  eigenvector
      for (size_t i = 0; i < N; i++)
	{
	  element = gsl_complex_rect(EigVecRealImag[ev*N + i], 0.);
	  gsl_matrix_complex_set(EigVec, i, ev, element);
	}

      // If complex pair
      if ((gsl_pow_2(EigValImag[ev]) > 1.e-12) && (ev + 1 < nev))
	{
	  // Add complex conjugate eigenvalue
	  element = gsl_complex_conjugate(gsl_vector_complex_get(EigVal, ev));
	  gsl_vector_complex_set(EigVal, ev + 1, element);

	  // Add imaginary part to eigenvector
	  for (size_t i = 0; i < N; i++)
	    {
	      element = gsl_complex_rect(GSL_REAL(gsl_matrix_complex_get(EigVec, i, ev)),
					 EigVecRealImag[(ev + 1)*N + i]);
	      gsl_matrix_complex_set(EigVec, i, ev, element);
	    }

	  // Add complex conjugate eigenvector
	  for (size_t i = 0; i < N; i++)
	    {
	      element = gsl_complex_conjugate(gsl_matrix_complex_get(EigVec, i, ev));
	      gsl_matrix_complex_set(EigVec, i, ev + 1, element);
	    }

	  // Increment eigenvalue one more time
	  ev++;
	}
    }

  //Free
  delete[] EigValReal;
  delete[] EigValImag;
  delete[] EigVecRealImag;

  return;
}


/**
 * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
 * \param[in] fEigVal    File descriptor for eigenvalues.
 * \param[in] fEigVec    File descriptor for eigenvectors.
 * \param[in] EigVal     Array of eigenvalues real parts.
 * \param[in] EigVec     Array of eigenvectors.
 */
void
writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		const gsl_vector_complex *EigVal,
		const gsl_matrix_complex *EigVec)
{
  // Print eigenvalues
  gsl_vector_complex_fprintf(fEigVal, EigVal, "%.12lf");
  
  /** Check for printing errors */
  if (ferror(fEigVal))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvalues");
  }

  // Print eigenvectors
  gsl_matrix_complex_fprintf(fEigVec, EigVec, "%.12lf");
  
  /** Check for printing errors */
  if (ferror(fEigVec))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvectors");
  }

  return;
}


#endif
