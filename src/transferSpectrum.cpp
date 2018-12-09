#include <transferSpectrum.hpp>
#include <cstring>

/** \file transferSpectrum.cpp
 *  \brief Definitions for transferSpectrum.hpp
 *
 */

/*
 * Global variables
 */
//! Default Arpack conf.
char methodStr[] = "LM";
configAR defaultCfgAR = {methodStr, 0, 0., 0, NULL, true}; 

/*
 * Constructors and destructors definitions
 */


/**
 * Constructor allocating space for a given number of eigenvalues and vectors
 * for a given transferOperator.
 * \param[in] nev_        Number of eigenvalues and eigenvectors
 * for which to allocate.
 * \param[in] transferOp_ Pointer to the transferOperator on which
 * to solve the eigen problem.
 * \param[in] cfgAR       Configuration data used by ARPACK++ for the
 * eigen problem.
 */
transferSpectrum::transferSpectrum(const int nev_,
				   const transferOperator *transferOp_,
				   const configAR cfgAR=defaultCfgAR)
  : N(transferOp_->getNFilled()), nev(nev_), transferOp(transferOp_),
    config(cfgAR), sorted(false),
    EigValForward(NULL), EigVecForward(NULL),
    EigValBackward(NULL), EigVecBackward(NULL) {}


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
 * Get left eigenvectors of transition matrix.
 */
void
transferSpectrum::getSpectrumForward()
{
  gsl_spmatrix *cpy;
  gsl_spmatrix2AR gsl2AR;
  /** Get transpose of transition matrix in CCS
   *  Transposing is trivial since the transition matrix is in CRS format.
   *  However, it is more secure to avoid directly changing the type of
   *  the transition matrix. */
  cpy = gsl_spmatrix_alloc_nzmax(transferOp->P->size2, transferOp->P->size1,
				 0, GSL_SPMATRIX_CCS);
  cpy->i = transferOp->P->i;
  cpy->data = transferOp->P->data;
  cpy->p = transferOp->P->p;
  cpy->nzmax = transferOp->P->nzmax;
  cpy->nz = transferOp->P->nz;

  gsl2AR = gsl_spmatrix2AR(cpy);

  //! Solve eigen value problem using user-defined ARPACK++
  getSpectrumAR(&nev, N, &EigProbForward, &gsl2AR, config, &EigValForward,
		&EigVecForward);

  // if (stationary) {
  //   /** Divide eigenvectors (real and imaginary parts)
  //    * by stationary distribution */
  //   for (size_t i = 0; i < N; i++) {
  //     if (gsl_vector_get(transferOp->initDist, i) > 0) {
  // 	element
  // 	  = gsl_complex_rect(1. / gsl_vector_get(transferOp->initDist, i), 0.);
  // 	view = gsl_matrix_complex_column(EigVecForward, i);
  // 	gsl_vector_complex_scale(&view.vector, element);
  //     }
  //   }
  // }

  return;
}


/**
 * Get right eigenvectors of transition matrix.
 */
void
transferSpectrum::getSpectrumBackward()
{
  gsl_spmatrix2AR gsl2AR;

  /** In the stationary case, the adjoint eigenvectors can be calculated
   *  as left eigenvectors of the forward transition matrix
   *  divided by the stationary distribution */
  gsl2AR = gsl_spmatrix2AR(transferOp->P);
  
  /** Get eigenvalues and vectors of backward transition matrix */
  getSpectrumAR(&nev, N, &EigProbBackward, &gsl2AR, config, &EigValBackward,
		&EigVecBackward);

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
 * where \f$\psi_i\f$ and \f$\psi_i^*\f$ are the forward
 * and backward (adjoint) eigenvectors,
 * respectively and norm is for \f$L^2\f$ w.r.t the initial distribution.
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
	= gsl_matrix_complex_const_row(EigVecForward, ev);
      normForward = gsl_vector_complex_get_norm(&viewFor.vector);

      //! Get norm of backward eigenvector.
      gsl_vector_complex_const_view viewBack
	= gsl_matrix_complex_const_row(EigVecBackward, ev);
      normBackward = gsl_vector_complex_get_norm(&viewBack.vector,
						 transferOp->initDist);

      //! Divide by their inner product (in case not made biorthonormal)
      inner
	= GSL_REAL(gsl_vector_complex_get_inner_product(&viewFor.vector,
							&viewBack.vector));

      //! Set condition number
      gsl_vector_set(conditionNumbers, ev,
		     normForward * normBackward / inner);
    }

  return conditionNumbers;
}


/**
 * Get weights associated with the projection of two observables
 * on the backward and forward eigenvectors, respectively, i.e.
 * \f$ w_i = \left<f, \psi_k^* \right>_{\rho_0}
 * \left<\psi_k, g \right>_{\rho_f} \f$.
 * \param[in] f Vector representing the first observable,
 * projected on the backward eigenvectors.
 * \param[in] g Vector representing the second observable,
 * projected on the forward eigenvectors.
 * \return Weights associated with each pair of eigenvectors
 * for the two observables.
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
	= gsl_matrix_complex_const_row(EigVecBackward, ev);
      weight = gsl_vector_complex_get_inner_product(fc, &viewBack.vector,
						    transferOp->initDist);

      // Project on forward eigenvector
      gsl_vector_complex_const_view viewFor
	= gsl_matrix_complex_const_row(EigVecForward, ev);
      weight = gsl_complex_mul(weight,
			       gsl_vector_complex_get_inner_product(&viewFor.vector, gc));

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
  gsl_matrix_complex *tmpEigVecFor = gsl_matrix_complex_alloc(nev, N);
  gsl_matrix_complex *tmpEigVecBack = gsl_matrix_complex_alloc(nev, N);
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
  gsl_permute_matrix_complex(sort_idx, EigVecForward, 0);

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
	gsl_matrix_complex_const_row(tmpEigVecBack, ev);
      gsl_matrix_complex_set_row(EigVecBackward, idx, &view.vector);
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
       std::cerr << "transferSpectrum::makeBiorthonormal,\
backward eigen problem solved first before to make biorthonormal set"
		 << std::endl;
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
   	 = gsl_matrix_complex_const_row(EigVecForward, ev);
       gsl_vector_complex_view viewBack
   	 = gsl_matrix_complex_row(EigVecBackward, ev);
       inner = gsl_vector_complex_get_inner_product(&viewFor.vector,
   						    &viewBack.vector);
       //! Divide backward eigenvector by conjugate of inner product
       inner = gsl_complex_conjugate(inner);
       inner = gsl_complex_inverse(inner);
       gsl_vector_complex_scale(&viewBack.vector, inner);
     }
   
   return;
}


/**
 * Write eigenvalues of forward transition matrix.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] fileFormat         String "bin" or "txt" for the output type.
 * \return                       Exit status.
 */
void
transferSpectrum::writeEigValForward(const char *EigValForwardFile,
				     const char *fileFormat) const
{
  FILE *streamEigVal;
  
  /** Open files for forward */
  if (!(streamEigVal = fopen(EigValForwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream for writing forward eigenvalues");
    }

  /** Write forward */
  writeSpectrumAR(streamEigVal, EigValForward, fileFormat);

  /** Close */
  fclose(streamEigVal);

  return;
}


/**
 * Write eigenvalues and of backward transition matrix.
 * \param[in] EigValBackwardFile  File name of the file to print backward eigenvalues.
 * \param[in] fileFormat          String "bin" or "txt" for the output type.
 * \return                        Exit status.
 */
void
transferSpectrum::writeEigValBackward(const char *EigValBackwardFile,
				      const char *fileFormat) const
{
  FILE *streamEigVal;
  
  /** Open files back backward */
  if (!(streamEigVal = fopen(EigValBackwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream back writing backward eigenvalues");
    }

  /** Write backward */
  writeSpectrumAR(streamEigVal, EigValBackward, fileFormat);

  /** Close */
  fclose(streamEigVal);

  return;
}


/**
 * Write eigenvalues and eigenvectors of forward transition matrix.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] EigVecForwardFile  File name of the file to print forward eigenvectors.
 * \param[in] fileFormat         String "bin" or "txt" for the output type.
 * \return                       Exit status.
 */
void
transferSpectrum::writeSpectrumForward(const char *EigValForwardFile,
				       const char *EigVecForwardFile,
				       const char *fileFormat="txt") const
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
  writeSpectrumAR(streamEigVal, streamEigVec, EigValForward, EigVecForward,
		  fileFormat);

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);

  return;
}


/**
 * Write eigenvalues and eigenvectors of backward transition matrix.
 * \param[in] EigValBackwardFile  File name of the file to print backward eigenvalues.
 * \param[in] EigVecBackwardFile  File name of the file to print backward eigenvectors.
 * \param[in] fileFormat          String "bin" or "txt" for the output type.
 * \return                        Exit status.
 */
void
transferSpectrum::writeSpectrumBackward(const char *EigValBackwardFile,
					const char *EigVecBackwardFile,
					const char *fileFormat="txt") const
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
  writeSpectrumAR(streamEigVal, streamEigVec, EigValBackward, EigVecBackward,
		  fileFormat);

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
 * \param[in] fileFormat         String "bin" or "txt" for the output type.
 * \return                       Exit status.
 */
void
transferSpectrum::writeSpectrum(const char *EigValForwardFile,
				const char *EigVecForwardFile,
				const char *EigValBackwardFile,
				const char *EigVecBackwardFile,
				const char *fileFormat="txt") const
{
  //! Write forward eigenvalues and eigenvectors
  writeSpectrumForward(EigValForwardFile, EigVecForwardFile,
		       fileFormat);

  //! Write backward eigenvalues and eigenvectors
  writeSpectrumBackward(EigValBackwardFile, EigVecBackwardFile,
			fileFormat);
  
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
      for (outerIdx = 0; outerIdx < M->size1; ++outerIdx)
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
      for (outerIdx = 0; outerIdx < M->size2; ++outerIdx)
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
 * \param[in]     N          Dimension of the square matrix.
 * \param[in,out] EigProb    Eigen problem to use.
 * \param[in]     gsl2AR     Object interfacing the GSL sparse matrix
 *                           to ARPACK++.
 * \param[in]     cfgAR      Configuration options passed as a configAR object.
 * \param[out]    EigVal     Found eigenvalues.
 * \param[out]    EigVec     Found eigenvectors.
*/
void
getSpectrumAR(int *nev, const size_t N,
	      ARNonSymStdEig<double, gsl_spmatrix2AR > *EigProb,
	      gsl_spmatrix2AR *gsl2AR, configAR cfgAR,
	      gsl_vector_complex **EigVal, gsl_matrix_complex **EigVec)
{
  gsl_complex element;

  // Allocate vectors given to ARPACK++
  double *EigValReal = new double [*nev+1];
  double *EigValImag = new double [*nev+1];
  double *EigVecRealImag = new double [(*nev+2) * N];

  //! Define non-hermitian eigenvalue problem
  EigProb->DefineParameters(gsl2AR->M->size1, *nev, gsl2AR,
			    &gsl_spmatrix2AR::MultMv,
			    cfgAR.which, cfgAR.ncv, cfgAR.tol,
			    cfgAR.maxit, cfgAR.resid, cfgAR.AutoShift);

  //! Find eigenvalues and left eigenvectors with ARPACK++
  EigProb->EigenValVectors(EigVecRealImag, EigValReal, EigValImag);

  /** Update number of eigenvalues */
  *nev = EigProb->ConvergedEigenvalues();
  
  /** Allocate eigenvalues and eigenvectors vectors with updated number */
  *EigVal = gsl_vector_complex_alloc(*nev);
  /** Row major */
  *EigVec = gsl_matrix_complex_alloc(*nev, N);

  //! Save eigenvalues and eigenvectors and their complex conjugate
  for (size_t ev = 0; ev < (size_t) *nev; ev++)
    {
      element = gsl_complex_rect(EigValReal[ev], EigValImag[ev]);
      gsl_vector_complex_set(*EigVal, ev, element);
      
      // Add real part of  eigenvector
      for (size_t i = 0; i < N; i++)
	{
	  element = gsl_complex_rect(EigVecRealImag[ev*N + i], 0.);
	  gsl_matrix_complex_set(*EigVec, ev, i, element);
	}

      // If complex pair
      if ((gsl_pow_2(EigValImag[ev]) > 1.e-12) && (ev + 1 < *nev))
	{
	  // Add complex conjugate eigenvalue
	  element = gsl_complex_conjugate(gsl_vector_complex_get(*EigVal, ev));
	  gsl_vector_complex_set(*EigVal, ev + 1, element);

	  // Add imaginary part to eigenvector
	  for (size_t i = 0; i < N; i++)
	    {
	      element = gsl_complex_rect(GSL_REAL(gsl_matrix_complex_get(*EigVec,
									 ev, i)),
					 EigVecRealImag[(ev + 1)*N + i]);
	      gsl_matrix_complex_set(*EigVec, ev, i, element);
	    }

	  // Add complex conjugate eigenvector
	  for (size_t i = 0; i < N; i++)
	    {
	      element = gsl_complex_conjugate(gsl_matrix_complex_get(*EigVec, ev, i));
	      gsl_matrix_complex_set(*EigVec, ev + 1, i, element);
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
 * Write complex eigenvalues obtained as arrays from ARPACK++.
 * \param[in] fEigVal    File descriptor for eigenvalues.
 * \param[in] EigVal     Array of eigenvalues real parts.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 */
void
writeSpectrumAR(FILE *fEigVal, const gsl_vector_complex *EigVal,
		const char *fileFormat)
{
  // Print eigenvalues
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_complex_fwrite(fEigVal, EigVal);
  else
    gsl_vector_complex_fprintf(fEigVal, EigVal, "%.12lf");
  
  /** Check for printing errors */
  if (ferror(fEigVal))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvalues");
  }

  return;
}

/**
 * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
 * \param[in] fEigVal    File descriptor for eigenvalues.
 * \param[in] fEigVec    File descriptor for eigenvectors.
 * \param[in] EigVal     Array of eigenvalues real parts.
 * \param[in] EigVec     Array of eigenvectors.
 * \param[in] fileFormat String "bin" or "txt" for the output type.
 */
void
writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		const gsl_vector_complex *EigVal,
		const gsl_matrix_complex *EigVec,
		const char *fileFormat="txt")
{
  // Print eigenvalues
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_complex_fwrite(fEigVal, EigVal);
  else
    gsl_vector_complex_fprintf(fEigVal, EigVal, "%.12lf");
  
  /** Check for printing errors */
  if (ferror(fEigVal))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvalues");
  }

  // Print eigenvectors
  if (strcmp(fileFormat, "bin") == 0)
    gsl_matrix_complex_fwrite(fEigVec, EigVec);
  else
    gsl_matrix_complex_fprintf(fEigVec, EigVec, "%.12lf");
  
  /** Check for printing errors */
  if (ferror(fEigVec))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvectors");
  }

  return;
}

// /**
//  * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
//  * For the binary file format, compression is applied:
//  * - Each eigenvalue is a given by its real part, imaginary part
//  *   and an integer giving its type:
//  *   + (0) real
//  *   + (1) complex conjugate with positive imaginary part
//  *   + (2) complex conjugate with negative imaginary part
//  * - Real eigenvectors (0) are written as usual
//  * - Complex conjugate eigenvectors
//  * \param[in] fEigVal    File descriptor for eigenvalues.
//  * \param[in] fEigVec    File descriptor for eigenvectors.
//  * \param[in] EigVal     Array of eigenvalues real parts.
//  * \param[in] EigVec     Array of eigenvectors.
//  * \param[in] fileFormat String "bin" or "txt" for the output type.
//  */
// void
// writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
// 		const gsl_vector_complex *EigVal,
// 		const gsl_matrix_complex *EigVec,
// 		const char *fileFormat="txt")
// {
//   // Print eigenvalues
//   if (strcmp(fileFormat, "bin") == 0)
//     for (size_t
//     gsl_vector_complex_fwrite(fEigVal, EigVal);
//   else
//     gsl_vector_complex_fprintf(fEigVal, EigVal, "%.12lf");
  
//   /** Check for printing errors */
//   if (ferror(fEigVal))
//     {
//       throw std::ios::failure("writeSpectrumAR, printing eigenvalues");
//   }

//   // Print eigenvectors
//   if (strcmp(fileFormat, "bin") == 0)
//     gsl_matrix_complex_fwrite(fEigVec, EigVec);
//   else
//     gsl_matrix_complex_fprintf(fEigVec, EigVec, "%.12lf");
  
//   /** Check for printing errors */
//   if (ferror(fEigVec))
//     {
//       throw std::ios::failure("writeSpectrumAR, printing eigenvectors");
//   }

//   return;
// }

