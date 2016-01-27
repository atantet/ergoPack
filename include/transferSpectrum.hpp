#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <gsl/gsl_spmatrix.h>
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
  bool AutoShift;    
} configAR;
/** Declare default structure looking for largest magnitude eigenvalues */
configAR defaultCfgAR = {"LM", 0, 0., 0, NULL, true};


/** \brief Transfer operator spectrum.
 *
 *  Class used to calculate the spectrum of forward and backward
 *  transition matrices of a transferOperator object.
 */
class transferSpectrum {
public:
  int nev = 0;                  //!< Number of eigenvalues and vectors to search;
  double *EigValForwardReal;    //!< Real part of eigenvalues of forward transition matrix
  double *EigValForwardImag;    //!< Imaginary part of eigenvalues of forward transition matrix
  double *EigValBackwardReal;   //!< Real part of eigenvalues of backward transition matrix
  double *EigValBackwardImag;   //!< Imaginary part of eigenvalues of backward transition matrix
  double *EigVecForward;        //!< Eigenvectors of forward transition matrix
  double *EigVecBackward;       //!< Eigenvectors of backward transition matrix
  transferOperator *transferOp; //!< Transfer operator of the eigen problem


  /** \brief Default constructor. */
  transferSpectrum(){}
  /** \brief Constructor allocating for nev_ eigenvalues and vectors. */
  transferSpectrum(const int nev_, transferOperator *transferOp_);
  /** \brief Destructor desallocating. */
  ~transferSpectrum();


  /** \brief Get transfer operator spectrum. */
  int getSpectrum(const configAR cfgAR=defaultCfgAR);
  

  /** \brief Write complex eigenvalues and eigenvectors from ARPACK++. */
  int writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
		    const char *EigValBackwardFile, const char *EigVecBackwardFile);
};


/** \brief Interface from GSL sparse matrices to ARPACK++
 */
template<class T>
class gsl_spmatrix2AR {
public:
  gsl_spmatrix *M; //!< GSL sparse matrix


  /** \brief Default constructor */
  gsl_spmatrix2AR(){ M = NULL; }
  /** \brief Constructor from GSL sparse matrix */
  gsl_spmatrix2AR(gsl_spmatrix *M_);

  
  /** \brief Matrix-vector product required by ARPACK */
  void MultMv(T *v, T *w);
};


/*
 *  Functions declarations
 */
/** \brief Get spectrum of a nonsymmetric matrix using ARPACK++. */
int getSpectrumAR(gsl_spmatrix2AR<double> *gsl2AR, int nev, configAR cfgAR,
		  double *EigValReal, double *EigValImag, double *EigVec);
void writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		     const double *EigValReal, const double *EigValImag,
		     const double *EigVec, const int nev, const size_t N);


/*
 * Constructors and destructors definitions
 */

/**
 * Constructor allocating space for a given number of eigenvalues and vectors
 * for a given transferOperator.
 * \param[in] nev_        Number of eigenvalues and eigenvectors for which to allocate.
 * \param[in] transferOp_ Pointer to the transferOperator on which to solve the eigen problem.
 */
transferSpectrum::transferSpectrum(const int nev_, transferOperator *transferOp_)
{
  /** Number of eigenvalues to search and transferOperator */
  nev = nev_;
  transferOp = transferOp_;
  
  /** Allocate for possible additional eigenvalue of complex pair */
  EigValForwardReal = new double [nev+1];
  EigValForwardImag = new double [nev+1];
  EigValBackwardReal = new double [nev+1];
  EigValBackwardImag = new double [nev+1];
  
  /** Allocate for both real and imaginary part
   *  but only for one member of the pair */
  EigVecForward = new double [(nev+2) * transferOp->getN()];
  EigVecBackward = new double [(nev+2) * transferOp->getN()];
}


/**
 * Destructor desallocates. 
 */
transferSpectrum::~transferSpectrum()
{
  delete[] EigValForwardReal;
  delete[] EigValForwardImag;
  delete[] EigVecForward;
  delete[] EigValBackwardReal;
  delete[] EigValBackwardImag;
  delete[] EigVecBackward;
}


/*
 * Methods definitions
 */

/**
 * Get spectrum of transfer operator,
 * including the complex eigenvalues,
 * the left eigenvectors of the forward transition matrix
 * and the right eigenvectors of the backward transitioin matrix.
 * The vectors of eigenvalues and eigenvectors should not be preallocated.
 */
int
transferSpectrum::getSpectrum(configAR cfgAR)
{
  gsl_spmatrix *cpy;
  gsl_spmatrix2AR<double> gsl2AR;
  int nconv = 0;
  
  /** Check if constructor has been called */
  if (!nev)
    {
      throw std::exception();
    }

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
  gsl2AR = gsl_spmatrix2AR<double>(cpy);

  /** Solve eigen value problem using user-defined ARPACK++ */
  nconv += getSpectrumAR(&gsl2AR, nev, cfgAR,
			 EigValForwardReal, EigValForwardImag, EigVecForward);

  
  /** Get transpose of backward transition matrix in ARPACK CCS format */
  cpy = gsl_spmatrix_alloc_nzmax(transferOp->Q->size2, transferOp->Q->size1, 0, GSL_SPMATRIX_CCS);
  cpy->innerSize = transferOp->Q->innerSize;
  cpy->outerSize = transferOp->Q->outerSize;
  cpy->i = transferOp->Q->i;
  cpy->data = transferOp->Q->data;
  cpy->p = transferOp->Q->p;
  cpy->nzmax = transferOp->Q->nzmax;
  cpy->nz = transferOp->Q->nz;
  gsl2AR = gsl_spmatrix2AR<double>(cpy);
  
  /** Get eigenvalues and vectors of backward transition matrix */
  nconv += getSpectrumAR(&gsl2AR, nev, cfgAR,
			 EigValBackwardReal, EigValBackwardImag, EigVecBackward);
    
  return nconv;
}
    

/**
 * Write complex eigenvalues and eigenvectors
 * of forward and backward transition matrices of a transfer operator to file.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] EigVecForwardFile  File name of the file to print forward eigenvectors.
 * \param[in] EigValBackwardFile File name of the file to print backward eigenvalues.
 * \param[in] EigVecBackwardFile File name of the file to print backward eigenvectors.
 * \return                       Exit status.
 */
int
transferSpectrum::writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
				const char *EigValBackwardFile, const char *EigVecBackwardFile)
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
  writeSpectrumAR(streamEigVal, streamEigVec,
		  EigValForwardReal, EigValForwardImag, EigVecForward,
		  nev, transferOp->getN());

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);

  /** Open files for backward */
  if (!(streamEigVal = fopen(EigValBackwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream for writing backward eigenvalues");
    }
  if (!(streamEigVec = fopen(EigVecBackwardFile, "w")))
    {
      throw std::ios::failure("transferSpectrum::writeSpectrum, \
opening stream for writing backward eigenvectors");
    }

  /** Write backward */
  writeSpectrumAR(streamEigVal, streamEigVec,
		  EigValBackwardReal, EigValBackwardImag, EigVecBackward,
		  nev, transferOp->getN());

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);
  
  return 0;
}


/**
 * Constructor for interface from GSL sparse matrices to ARPACK++.
 * \param[in] M_ GSL sparse matrix from which the spectrum will be calculated.
 */
template <class T>
gsl_spmatrix2AR<T>::gsl_spmatrix2AR(gsl_spmatrix *M_)
{
  M = M_;
}


/**
 * Matrix-vector product required by ARPACK++ when using user-defined matrices
 * (GSL sparse matrices). Modified version of gsl_spblas_dgemv in gsl_spmatrix.h .
 * \param[in]  v Vector to multiply.
 * \param[out] w Vector storing the result of multiplication.
 */
template <class T>
void
gsl_spmatrix2AR<T>::MultMv(T *v, T *w)
{
  size_t j, outerIdx, p, n;
  
  /* form y := 0 */
  for (j = 0; j < M->size1; ++j)
    w[j] = 0.0;

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
 * \param[in]  gsl2AR     Object interfacing the GSL sparse matrix to ARPACK++.
 * \param[in]  nev        Number of eigenvalues and eigenvectors to find.
 * \param[in]  cfgAR      Configuration options passed as a configAR object.
 * \param[out] EigValReal Real part of found eigenvalues.
 * \param[out] EigValImag Imaginary  part of found eigenvalues.
 * \param[out] EigVec     Found eigenvectors.
 * \return                Number of eigenvalues and eigenvectors found.
 */
int
getSpectrumAR(gsl_spmatrix2AR<double> *gsl2AR, int nev, configAR cfgAR,
	      double *EigValReal, double *EigValImag, double *EigVec)
{
  int nconv;
  
  // Define eigen problem
  ARNonSymStdEig<double, gsl_spmatrix2AR<double> > EigProb(gsl2AR->M->size1, nev, gsl2AR,
							   &gsl_spmatrix2AR<double>::MultMv,
							   cfgAR.which, cfgAR.ncv, cfgAR.tol,
							   cfgAR.maxit, cfgAR.resid,
							   cfgAR.AutoShift);
  
  // Find eigenvalues and left eigenvectors
  EigProb.EigenValVectors(EigVec, EigValReal, EigValImag);
  nconv = EigProb.ConvergedEigenvalues();


  return nconv;
}


/**
 * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
 * \param[in] fEigVal    File descriptor for eigenvalues.
 * \param[in] fEigVec    File descriptor for eigenvectors.
 * \param[in] EigValReal Array of eigenvalues real parts.
 * \param[in] EigValImag Array of eigenvalues imaginary parts.
 * \param[in] EigVec     Array of eigenvectors.
 * \param[in] nev        Number of eigenvalues and eigenvectors.
 * \param[in] N          Length of the eigenvectors.
 */
void
writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		const double *EigValReal, const double *EigValImag,
		const double *EigVec, const int nev, const size_t N)
{
  size_t vecCount = 0;
  int ev =0;
  // Write real and imaginary parts of each eigenvalue on each line
  // Write on each pair of line the real part of an eigenvector then its imaginary part
  while (ev < nev) {
    // Always write the eigenvalue
    fprintf(fEigVal, "%lf %lf\n", EigValReal[ev], EigValImag[ev]);
    // Always write the real part of the eigenvector ev
    for (size_t i = 0; i < N; i++){
      fprintf(fEigVec, "%lf ", EigVec[vecCount*N+i]);
    }
    fprintf(fEigVec, "\n");
    vecCount++;
    
    // Write its imaginary part or the zero vector
    if (EigValImag[ev] != 0.){
      for (size_t i = 0; i < N; i++)
	fprintf(fEigVec, "%lf ", EigVec[vecCount*N+i]);
      vecCount++;
      // Skip the conjugate
      ev += 2;
    }
    else{
      for (size_t i = 0; i < N; i++)
	fprintf(fEigVec, "%lf ", 0.);
      ev += 1;
    }
    fprintf(fEigVec, "\n");
  }

  /** Check for printing errors */
  if (ferror(fEigVal))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvalues");
  }
  if (ferror(fEigVec))
    {
      throw std::ios::failure("writeSpectrumAR, printing eigenvectors");
  }

  return;
}
	
#endif
