#ifndef GSL_MATRIX_EXTENSION_H
#define GSL_MATRIX_EXTENSION_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <math.h>

/** \file gsl_extension.h
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

/** \brief Get the sum of the elements of a matrix over each row. */
gsl_vector *gsl_matrix_get_rowsum(const gsl_matrix *m);
/** \brief Get the sum of the elements of a matrix over each column. */
gsl_vector *gsl_matrix_get_colsum(const gsl_matrix *m);
/** \brief Get the mean over the elements of a matrix along a given axis. */
gsl_vector *gsl_matrix_get_mean(const gsl_matrix *m, const size_t axis);
/** \brief Get the variance over the elements of a matrix along a given axis. */
gsl_vector *gsl_matrix_get_variance(const gsl_matrix *m, const size_t axis);
/** \brief Get the standard deviation over the elements of a matrix along a given axis. */
gsl_vector *gsl_matrix_get_std(const gsl_matrix *m, const size_t axis);


/*
 * Definitions
 */

/** 
 * Get the square root of the elements of a vector.
 * \param[in/out] v Vector for which to calculate the square root.
 * \return          Exit status.
 */
int
gsl_vector_sqrt(gsl_vector *v)
{
  size_t i;
  
  for (i = 0; i < v->size; i++)
    v->data[i * v->stride] = sqrt(v->data[i * v->stride]);
      
  return GSL_SUCCESS;
}

/**
 * Normalize vector by the sum of its elements.
 * \param[input] v Vector to normalize.
 */
int
gsl_vector_normalize(gsl_vector *v)
{
  // Get sum of elements
  double sum = gsl_vector_get_sum(v);

  // Divide by sum of elements
  gsl_vector_scale(v, 1. / sum);

  return GSL_SUCCESS;
}

/**
 * Get sum of vector elements.
 * \param[input] v Vector from which to sum the elements.
 * \return         Sum of vector elements.
 */
double
gsl_vector_get_sum(const gsl_vector *v)
{
  double sum = 0;

  for (size_t j = 0; j < v->size; j++)
    sum += v->data[j * v->stride];
  
  return sum;
}


/**
 * Get product of vector elements.
 * \param[input] v Vector from which to multiply the elements.
 * \return         Product of vector elements.
 */
double
gsl_vector_get_prod(const gsl_vector *v)
{
  double prod = 1.;

  for (size_t j = 0; j < v->size; j++)
    prod *= v->data[j * v->stride];
  
  return prod;
}


/**
 * Get product of vector elements.
 * \param[input] v Vector from which to multiply the elements.
 * \return         Product of vector elements.
 */
double
gsl_vector_uint_get_prod(const gsl_vector_uint *v)
{
  double prod = 1.;

  for (size_t j = 0; j < v->size; j++)
    prod *= v->data[j * v->stride];
  
  return prod;
}


/** 
 * Get the mean of the elements of a vector
 * \param[in] v    Vector over which to calculate the mean.
 * \return         Mean of the vector.
 */
double
gsl_vector_get_mean(const gsl_vector *v)
{
  return gsl_vector_get_sum(v) / v->size;
}


/** 
 * Get the variance of the elements of a vector
 * \param[in] v    Vector over which to calculate the variance.
 * \return         Variance of the vector.
 */
double
gsl_vector_get_var(const gsl_vector *v)
{
  double var, mean;
  gsl_vector *v2 = gsl_vector_alloc(v->size);

  // Get vector squared
  gsl_vector_memcpy(v2, v);
  gsl_vector_mul(v2, v);

  // Get mean of squares
  var = gsl_vector_get_mean(v2);

  // Remove square of mean
  mean = gsl_vector_get_mean(v);
  var -= gsl_pow_2(mean);

  // Free
  gsl_vector_free(v2);

  return var;
}


/** 
 * Get the standard deviation of the elements of a vector
 * \param[in] v    Vector over which to calculate the standard deviation.
 * \return         Standard Deviation of the vector.
 */
double
gsl_vector_get_std(const gsl_vector *v)
{
  return sqrt(gsl_vector_get_var(v));
}


/**
 * Get index of maximum of vector among marked elements.
 * \param[in] v      Vector for which to find the maximum.
 * \param[in] marker Boolean vector marking the elements among which
 *                   to look for the maximum.
 * \return           Index of the maximum.
 */
size_t gsl_vector_max_index_among(const gsl_vector *v, const gsl_vector_uint *marker)
{
  size_t max_index;
  double max;
  size_t i;

  if (v->size != marker->size)
    GSL_ERROR("Vector and marker should have the same size.", GSL_EINVAL);

  // Get value of first marked element
  i = 0;
  while (!gsl_vector_uint_get(marker, i))
    i++;
  max = gsl_vector_get(v, i);
  max_index = i;

  // Get maximum  
  for (i = 0; i < v->size; i++)
    {
      if (gsl_vector_uint_get(marker, i)
	  && (max < gsl_vector_get(v, i)))
	{
	  max = gsl_vector_get(v, i);
	  max_index = i;
	}
    }
  
  return max_index;
}

/**
 * Get vector of a given number equally-spaced numbers between two bounds.
 * \param[out] v     Vector for which to set the elements.
 * \param[in]  lower Lower bound.
 * \param[in]  upper Upper bound.
 */
int
gsl_vector_linspace(gsl_vector *v, const double lower, const double upper)
{
  size_t i;
  double delta = (upper - lower) / (v->size - 1);
  
  for (i = 0; i < v->size; i++)
    {
      gsl_vector_set(v, i, lower + i * delta);
    }
  
  return GSL_SUCCESS;
}







/** 
 * Get the sum of the elements of a matrix over each row.
 * \param[in] m Matrix over which to sum.
 * \return      Vector of the sum of the rows.
 */
gsl_vector *
gsl_matrix_get_rowsum(const gsl_matrix *m)
{
  size_t i;
  gsl_vector *sum = gsl_vector_alloc(m->size1);
  
  for (i = 0; i < m->size1; i++)
    {
      gsl_vector_const_view view = gsl_matrix_const_row(m, i);
      gsl_vector_set(sum, i, gsl_vector_get_sum(&view.vector));
    }

  return sum;
}
  

/** 
 * Get the sum of the elements of a matrix over each column.
 * \param[in] m Matrix over which to sum.
 * \return      Vector of the sum of the columns.
 */
gsl_vector *
gsl_matrix_get_colsum(const gsl_matrix *m)
{
  size_t j;
  gsl_vector *sum = gsl_vector_alloc(m->size2);
  
  for (j = 0; j < m->size2; j++)
    {
      gsl_vector_const_view view = gsl_matrix_const_column(m, j);
      gsl_vector_set(sum, j, gsl_vector_get_sum(&view.vector));
    }

  return sum;
}


/** 
 * Get the mean of the elements of a matrix along a given axis.
 * \param[in] m    Matrix over which to calculate the mean.
 * \param[in] axis Axis over which to calculate the mean
 *                 (0: along columns, 1: along rows).
 * \return         Vector of with the means.
 */
gsl_vector *
gsl_matrix_get_mean(const gsl_matrix *m, const size_t axis)
{
  gsl_vector *mean;

  switch (axis)
    {
    case 0:
      mean = gsl_matrix_get_colsum(m);
      gsl_vector_scale(mean, 1. / m->size1);
      break;
    case 1:
      mean = gsl_matrix_get_rowsum(m);
      gsl_vector_scale(mean, 1. / m->size2);
      break;
    default:
      GSL_ERROR_NULL("axis should be 0 or 1.", GSL_EINVAL);
    }

  return mean;
}
  

/** 
 * Get the variance of the elements of a matrix along a given axis.
 * \param[in] m    Matrix over which to calculate the variance.
 * \param[in] axis Axis over which to calculate the variance
 *                 (0: along columns, 1: along rows).
 * \return         Vector of with the variance.
 */
gsl_vector *
gsl_matrix_get_var(const gsl_matrix *m, const size_t axis)
{
  gsl_vector *var, *mean;
  gsl_matrix *m2 = gsl_matrix_alloc(m->size1, m->size2);

  // Get matrix of squared elements
  gsl_matrix_memcpy(m2, m);
  gsl_matrix_mul_elements(m2, m);

  // Get mean of squares
  var = gsl_matrix_get_mean(m2, axis);

  // remove mean
  mean = gsl_matrix_get_mean(m, axis);
  gsl_vector_mul(mean, mean);
  gsl_vector_sub(var, mean);
  
  // Free
  gsl_matrix_free(m2);
  gsl_vector_free(mean);
  
  return var;
}
  
/** 
 * Get the standard deviation of the elements of a matrix along a given axis.
 * \param[in] m    Matrix over which to calculate the standard deviation.
 * \param[in] axis Axis over which to calculate the standard deviation
 *                 (0: along columns, 1: along rows).
 * \return         Vector of with the standard deviation.
 */
gsl_vector *
gsl_matrix_get_std(const gsl_matrix *m, const size_t axis)
{
  gsl_vector *std;

  // Get variance
  std = gsl_matrix_get_var(m, axis);

  // Get square root
  gsl_vector_sqrt(std);
  
  return std;
}
  
#endif
