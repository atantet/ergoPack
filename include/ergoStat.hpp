#ifndef ERGOSTAT_HPP
#define ERGOSTAT_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>


/** \file ergoStat.hpp
 *  \brief Various statistical routines.
 *  
 *  Various statistical routines used in ergoPack.
 */


/*
 * Function declarations
 */

//!< \brief EOF analysis (or PCA) of a multivariate time series.
void getEOF(const gsl_matrix *data, gsl_vector *w, gsl_matrix *E, gsl_matrix *A);

  
/*
 * Function definitions
 */

/**
 * Empirical Orthogonal Functions analysis (or Principal Component Analysis)
 * of a multivariate time series.
 * \param[in]  data Multivariate time series on which to perform the analysis.
 * \param[out] w    Eigenvalues of the covariance matrix giving the explained variance.
 * \param[out] E    Matrix with an Empirical Orthogonal Function for each column.
 * \param[out] A    Matrix with a principal component for each column.
 */
void
getEOF(const gsl_matrix *data, gsl_vector *w, gsl_matrix *E, gsl_matrix *A)
{
  size_t nt = data->size1;
  size_t N = data->size2;
  gsl_vector *mean;
  gsl_matrix *C = gsl_matrix_alloc(N, N);
  gsl_eigen_symmv_workspace *work = gsl_eigen_symmv_alloc(N);
  gsl_matrix *X = gsl_matrix_alloc(data->size1, data->size2);
  gsl_matrix_memcpy(X, data);
  gsl_vector_view col;

  // Get anomalies
  A = gsl_matrix_alloc(nt, N);
  gsl_matrix_memcpy(A, X);
  mean = gsl_matrix_get_mean(A, 0);
  for (size_t j = 0; j < X->size2; j++)
    {
      col = gsl_matrix_column(X, j);
      gsl_vector_add_constant(&col.vector, - gsl_vector_get(mean, j));
    }

  // Get correlation matrix
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1., A, A, 0., C);
  gsl_matrix_scale(C, 1. / nt);

  // Solve eigen problem and sort by descending magnitude
  gsl_eigen_symmv(C, w, E, work);
  gsl_eigen_symmv_sort(w, E, GSL_EIGEN_SORT_VAL_DESC);

  // Get principal components
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., X, E, 0., A);

  // Free
  gsl_eigen_symmv_free(work);
  gsl_matrix_free(C);

  return;
}


#endif
