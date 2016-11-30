#ifndef GSL_EXTENSION_HPP
#define GSL_EXTENSION_HPP

#include <cstdio>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_errno.h>
#include <math.h>

/** \file gsl_extension.hpp
 *  \brief Extend GSL with utility functions.
 *   
 *  Extend GSL with utility functions.
 */


/*
 * Declarations
 */

/** \brief Get the square root of the elements of a vector. */
int gsl_vector_sqrt(gsl_vector *v);
/** \brief Normalize vector by the sum of its elements. */
int gsl_vector_normalize(gsl_vector *v);
/** \brief Get sum of vector elements. */
double gsl_vector_get_sum(const gsl_vector *v);
/** \brief Get sum of squares of vector elements. */
double gsl_vector_get_sum_squares(const gsl_vector *v);
/** \brief Get inner product of two vectors for a given measure. */
double gsl_vector_get_innerproduct(const gsl_vector *v, const gsl_vector *w,
				   const gsl_vector *mu);
/** \brief Get \f$L^2\f$ norm of a vector for a given measure. */
double gsl_vector_get_norm(const gsl_vector *v, const gsl_vector *mu);
/** \brief Get product of vector elements. */
double gsl_vector_get_prod(const gsl_vector *v);
/** \brief Get product of vector elements. */
double gsl_vector_uint_get_prod(const gsl_vector_uint *v);
/** \brief Get the mean over the elements of a vector. */
double gsl_vector_get_mean(const gsl_vector *v);
/** \brief Get the variance over the elements of a vector. */
double gsl_vector_get_var(const gsl_vector *v);
/** \brief Get the standard deviation over the elements of a vector. */
double gsl_vector_get_std(const gsl_vector *v);
/** \brief Get index of maximum of vector among marked elements. */
size_t gsl_vector_max_index_among(gsl_vector *v, gsl_vector_uint *marker);
/** \brief Get vector of a given number equally-spaced numbers between two bounds. */
int gsl_vector_linspace(gsl_vector *v, const double lower, const double upper,
			const size_t n);


/** \brief Copy vector to real part of complex vector. */
int gsl_vector_complex_memcpy_real(gsl_vector_complex *dst, const gsl_vector *src);
/** \brief Copy vector to imaginary part of complex vector. */
int gsl_vector_complex_memcpy_imag(gsl_vector_complex *dst, const gsl_vector *src);
/** \brief Scale a complex vector with a real number. */
void gsl_vector_complex_scale_real(gsl_vector_complex *v, const double a);
/** \brief Get complex logarithm of complex vector. */
void gsl_vector_complex_log(gsl_vector_complex *res, const gsl_vector_complex *v);
/** \brief Get absolute values of complex vector. */
gsl_vector *gsl_vector_complex_abs(const gsl_vector_complex *v);
/** \brief Find the maximum in absolute value of a complex vector. */
gsl_complex gsl_vector_complex_max(const gsl_vector_complex *v);
/** \brief Find the index of the maximum in absolute value of a complex vector. */
size_t gsl_vector_complex_max_index(const gsl_vector_complex *v);
/** \brief Find the minimum in absolute value of a complex vector. */
gsl_complex gsl_vector_complex_min(const gsl_vector_complex *v);
/** \brief Find the index of the minimum in absolute value of a complex vector. */
size_t gsl_vector_complex_min_index(const gsl_vector_complex *v);
/** \brief Calculate inner product between two complex vectors for a given measure. */
gsl_complex gsl_vector_complex_get_inner_product(const gsl_vector_complex *v,
						 const gsl_vector_complex *w,
						 const gsl_vector *mu);
/** \brief Calculate the norm \f$L^2_\mu\f$ of a complex vector for a given measure \f$\mu\f$. */
double gsl_vector_complex_get_norm(const gsl_vector_complex *v, const gsl_vector *mu);
/** \brief Permute rows or columns of a matrix. */
int gsl_permute_matrix_complex(const gsl_permutation * p, gsl_matrix_complex * m,
			       const size_t axis);



/** \brief Get the sum of the elements of a matrix over each row. */
int gsl_matrix_get_rowsum(gsl_vector *sum, const gsl_matrix *m);
/** \brief Get the sum of the elements of a matrix over each column. */
int gsl_matrix_get_colsum(gsl_vector *sum, const gsl_matrix *m);
/** \brief Get the mean over the elements of a matrix along a given axis. */
int gsl_matrix_get_mean(gsl_vector *mean, const gsl_matrix *m, const size_t axis);
/** \brief Get the variance over the elements of a matrix along a given axis. */
int gsl_matrix_get_variance(gsl_vector *var, const gsl_matrix *m, const size_t axis);
/** \brief Get the standard deviation over the elements of a matrix along a given axis. */
int gsl_matrix_get_std(gsl_vector *std, const gsl_matrix *m, const size_t axis);
/** \brief Get the min over the elements of a matrix along a given axis. */
int gsl_matrix_get_min(gsl_vector *min, const gsl_matrix *m, const size_t axis);
/** \brief Get the max over the elements of a matrix along a given axis. */
int gsl_matrix_get_max(gsl_vector *max, const gsl_matrix *m, const size_t axis);

/** \brief Get the sum of the elements of a compressed matrix over each row. */
int gsl_spmatrix_get_rowsum(gsl_vector *sum, const gsl_spmatrix *m);
/** \brief Get the sum of the elements of a compressed matrix over each column. */
int gsl_spmatrix_get_colsum(gsl_vector *sum, const gsl_spmatrix *m);
/** \brief Get the sum of the elements of a sparse matrix. */
double gsl_spmatrix_get_sum(const gsl_spmatrix *m);
/** \brief Divide each row of a compressed matrix by a vector. */
int gsl_spmatrix_div_rows(gsl_spmatrix *m, const gsl_vector *v, const double tol);
/** \brief Divide each column of a compressed matrix by a vector. */
int gsl_spmatrix_div_cols(gsl_spmatrix *m, const gsl_vector *v, const double tol);
/** \brief Pre-allocate a sparse matrix to read from a binary stream. */
gsl_spmatrix *gsl_spmatrix_alloc2read(FILE *stream, const size_t type);

#endif
