#include <gsl_extension.hpp>

/** \file gsl_extension.cpp
 *  \brief Definitions for gsl_extension.hpp
 *
 */

/*
 * Definitions
 */

/** 
 * Get the square root of the elements of a vector.
 * \param[in,out] v Vector for which to calculate the square root.
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
 * \param[in] v Vector to normalize.
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
 * \param[in] v Vector from which to sum the elements.
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
 * Get the inner product of two vectors for a given measure.
 * The inner product of \f$L^2_\mu\f$ between \f$v\f$ and \f$w\f$ is equal to
 * \f$\sum_{k = 1}^n v_k \mu_k w_k \f$.
 * \param[in] v  Left vector (bra).
 * \param[in] w  Right vector (ket).
 * \param[in] mu Vector of the measure associated with the inner product.
 * \return       Inner product.
 */
double
gsl_vector_get_inner_product(const gsl_vector *v, const gsl_vector *w,
			     const gsl_vector *mu)
{
  double inner;
  gsl_vector *tmp;

  if (v->size != w->size)
    GSL_ERROR("Vectors and measure should have the same size.", GSL_EINVAL);

  if (mu)
    if (v->size != mu->size)
      GSL_ERROR("Vectors and measure should have the same size.", GSL_EINVAL);

  // Get the inner product
  tmp = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(tmp, v);
  // If mu is given, get the Euclidean mu-innerproduct
  // Otherwise, get the Euclidean innerproduct.
  if (mu)
     gsl_vector_mul(tmp, mu);
  gsl_vector_mul(tmp, w);
  inner = gsl_vector_get_sum(tmp);

  // Free
  gsl_vector_free(tmp);
  
  return inner;
}


/**
 * Get the \f$L^2_\mu\f$ norm of a vector for a given measure \f$\mu\f$.
 * The \f$L^2_\mu\f$ norm of \f$v\f$ is equal to
 * \f$\sqrt{\sum_{k = 1}^n v_k^2 \mu_k}\f$.
 * \param[in] v  Vector of which to calculate the norm.
 * \param[in] mu Vector of the measure associated with the norm.
 * \return       Norm of the vector.
 */
double
gsl_vector_get_norm(const gsl_vector *v, const gsl_vector *mu)
{
  double norm;
  gsl_vector *w;

  if (mu)
    if (v->size != mu->size)
      GSL_ERROR("Vector and measure should have the same size.", GSL_EINVAL);

  // Copy vector
  w = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(w, v);
    
  // Get the norm
  norm = sqrt(gsl_vector_get_inner_product(v, w, mu));

  // Free
  gsl_vector_free(w);
  
  return norm;
}


/**
 * Get product of vector elements.
 * \param[in] v Vector from which to multiply the elements.
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
 * \param[in] v Vector from which to multiply the elements.
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
size_t
gsl_vector_max_index_among(const gsl_vector *v, const gsl_vector_uint *marker)
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
    gsl_vector_set(v, i, lower + i * delta);
  
  return GSL_SUCCESS;
}


/**
 * Scale a complex vector with a real number.
 * \param[in/out] v Vector to scale.
 * \param[in]     a Real number with which to scale.
 */
void
gsl_vector_complex_scale_real(gsl_vector_complex *v, const double a)
{
  for (size_t i = 0; i < v->size; i++)
    gsl_vector_complex_set(v, i,
			   gsl_complex_mul_real(gsl_vector_complex_get(v, i), a));
  
  return;
}

/** 
 * Get complex logarithm of complex vector.
 * \param[out] res Complex vector in which to save the logaritm.
 * \param[in]  v   Complex vector from which to calculate the logarithm.
 */
void
gsl_vector_complex_log(gsl_vector_complex *res, const gsl_vector_complex *v)
{
  for (size_t i = 0; i < v->size; i++)
    gsl_vector_complex_set(res, i, gsl_complex_log(gsl_vector_complex_get(v, i)));
      
  return;
}
/** 
 * Copy vector to real part of complex vector.
 * \param[out] dst Destination complex vector to which copy real part.
 * \param[in]  src Source vector to copy.
 */
int
gsl_vector_complex_memcpy_real(gsl_vector_complex *dst, const gsl_vector *src)
{
  gsl_vector_view view;
  
  if (dst->size != src->size)
    GSL_ERROR("Vectors should have the same size.", GSL_EINVAL);

  //! Copy source vector to real part of complex destination vector
  view = gsl_vector_complex_real(dst);
  gsl_vector_add(&view.vector, src);
  
  return GSL_SUCCESS;
}


/** 
 * Copy vector to imaginary part of complex vector.
 * \param[out] dst Destination complex vector to which copy imaginary part.
 * \param[in]  src Source vector to copy.
 */
int
gsl_vector_complex_memcpy_imag(gsl_vector_complex *dst, const gsl_vector *src)
{
  gsl_vector_view view;
  
  if (dst->size != src->size)
    GSL_ERROR("Vectors should have the same size.", GSL_EINVAL);

  //! Copy source vector to imaginary part of complex destination vector
  view = gsl_vector_complex_imag(dst);
  gsl_vector_add(&view.vector, src);
  
  return GSL_SUCCESS;
}


/** 
 * Get absolute values of complex vector.
 * \param[in] v Complex vector for which to calculate the absolute value.
 * \return      Real vector of absolute values.
 */
gsl_vector *gsl_vector_complex_abs(const gsl_vector_complex *v)
{
  gsl_vector *abs = gsl_vector_alloc(v->size);

  for (size_t i = 0; i < v->size; i++)
    {
      gsl_vector_set(abs, i, gsl_complex_abs(gsl_vector_complex_get(v, i)));
    }
  
  return abs;
}


/**
 * Find the maximum in absolute value of a complex vector.
 * \param[in] v Complex vector from which to find the maximum.
 * return       Maximum in absolute value of a complex vector.
 */
gsl_complex
gsl_vector_complex_max(const gsl_vector_complex *v)
{
  gsl_complex max, element;
  double absMax;

  //! Find maximum
  max = gsl_vector_complex_get(v, 0);
  absMax = gsl_complex_abs(max);
  for (size_t i = 1; i < v->size; i++)
    {
      element = gsl_vector_complex_get(v, i);

      // Test if new maximum and update
      if (absMax < gsl_complex_abs(element))
	{
	  max = element;
	  absMax = gsl_complex_abs(max);
	}
    }

  return max;
}


/**
 * Find the index of the maximum in absolute value of a complex vector.
 * \param[in] v Complex vector from which to find the index of maximum.
 * return       Index of the maximum in absolute value of the complex vector.
 */
size_t
gsl_vector_complex_max_index(const gsl_vector_complex *v)
{
  size_t idx;
  gsl_complex max, element;
  double absMax;

  //! Find maximum
  idx = 0;
  max = gsl_vector_complex_get(v, idx);
  absMax = gsl_complex_abs(max);
  for (size_t i = 1; i < v->size; i++)
    {
      element = gsl_vector_complex_get(v, i);

      // Test if new maximum and update
      if (absMax < gsl_complex_abs(element))
	{
	  max = element;
	  absMax = gsl_complex_abs(max);
	  idx = i;
	}
    }

  return idx;
}


/**
 * Find the minimum in absolute value of a complex vector.
 * \param[in] v Complex vector from which to find the minimum.
 * return       Minimum in absolute value of a complex vector.
 */
gsl_complex
gsl_vector_complex_min(const gsl_vector_complex *v)
{
  gsl_complex min, element;
  double absMin;

  //! Find minimum
  min = gsl_vector_complex_get(v, 0);
  absMin = gsl_complex_abs(min);
  for (size_t i = 1; i < v->size; i++)
    {
      element = gsl_vector_complex_get(v, i);
      
      // Test if new minimum and update
      if (absMin > gsl_complex_abs(element))
	{
	  min = element;
	  absMin = gsl_complex_abs(min);
	}
    }
  
  return min;
}


/**
 * Find the index of the minimum in absolute value of a complex vector.
 * \param[in] v Complex vector from which to find the index of minimum.
 * return       Index of the minimum in absolute value of the complex vector.
 */
size_t
gsl_vector_complex_min_index(const gsl_vector_complex *v)
{
  size_t idx;
  gsl_complex min, element;
  double absMin;
  
  //! Find minimum
  idx = 0;
  min = gsl_vector_complex_get(v, idx);
  absMin = gsl_complex_abs(min);
  for (size_t i = 1; i < v->size; i++)
    {
      element = gsl_vector_complex_get(v, i);
      
      // Test if new minimum and update
      if (absMin > gsl_complex_abs(element))
	{
	  min = element;
	  absMin = gsl_complex_abs(min);
	  idx = i;
	}
    }

  return idx;
}


/**
 * Calculate inner product between two complex vectors for a given measure.
 * The inner product of \f$L^2_\mu\f$ between \f$v\f$ and \f$w\f$ is equal to
 * \f$\sum_{k = 1}^n v_k \mu_k w^*_k \f$.
 * \param[in] v  Left vector (bra).
 * \param[in] w  Right vector (ket).
 * \param[in] mu Measure with respect to which to calculate the inner product.
 * \return       Inner product between the two vectors.
 */
gsl_complex
gsl_vector_complex_get_inner_product(const gsl_vector_complex *v,
				     const gsl_vector_complex *w,
				     const gsl_vector *mu)
{
  gsl_complex inner = gsl_complex_rect(0., 0.);
  gsl_complex tmp;

  if ((v->size != w->size) || (v->size != mu->size))
    {
      fprintf(stderr, "Vectors and measure should have the same size.\n");
      return inner;
    }
  
  
  for (size_t i = 0; i < v->size; i++)
    {
      //! Get product i
      tmp = gsl_vector_complex_get(v, i);
      tmp = gsl_complex_mul_real(tmp, gsl_vector_get(mu, i));
      tmp = gsl_complex_mul(tmp, gsl_complex_conjugate(gsl_vector_complex_get(w, i)));

      //! Add product i	
      inner = gsl_complex_add(inner, tmp);
    }
  
  return inner;
}


/** 
 * Calculate the norm \f$L^2_\mu\f$ of a complex vector for a given measure \f$\mu\f$.
 * \param[in] v  Vector from which to calculate the norm.
 * \param[in] mu Measure with respect to which to calculate the norm.
 * \return       Norm of the vector.
 */
double
gsl_vector_complex_get_norm(const gsl_vector_complex *v, const gsl_vector *mu)
{
  gsl_complex norm;
  gsl_vector_complex *w;

  if (v->size != mu->size)
    GSL_ERROR("Vector and measure should have the same size.", GSL_EINVAL);

  //! Copy vector
  w = gsl_vector_complex_alloc(v->size);
  gsl_vector_complex_memcpy(w, v);

  //! Get norm
  norm = gsl_vector_complex_get_inner_product(v, w, mu);

  //! Free
  gsl_vector_complex_free(w);

  return sqrt(GSL_REAL(norm));
}


/**
 * Permute rows (axis = 0) or columns (axis = 1) of a matrix according to permutation.
 * \param[in]     p    Permutation.
 * \param[in,out] m    Matrix to permute.
 * \param[in]     axis Axis along which to permute.
 * \return             Exit status.
 */
int gsl_permute_matrix_complex(const gsl_permutation * p, gsl_matrix_complex * m,
			       const size_t axis)
{
  size_t sizeAxis;

  //! Get size of axis along which to permute
  if (axis == 0)
    {
      sizeAxis = m->size1;
    }
  else if (axis == 1)
    {
      sizeAxis = m->size2;
    }
  else
    {
      GSL_ERROR("Axis must be either 0 (along rows) or 1 (along columns).",
		GSL_EINVAL);
    }

  if (p->size != sizeAxis)
    {
      GSL_ERROR("Permutation must have the same size as axis of matrix along which\
to permute", GSL_EINVAL);
    }
  
  //! Save matrix
  gsl_matrix_complex *tmp = gsl_matrix_complex_alloc(m->size1, m->size2);
  gsl_matrix_complex_memcpy(tmp, m);

  //! Permute
  for (size_t i = 0; i < sizeAxis; i++)
    {
      if (axis == 0)
	{
	  gsl_vector_complex_const_view view
	    = gsl_matrix_complex_const_row(tmp, (gsl_permutation_data(p))[i]);
	  gsl_matrix_complex_set_row(m, i, &view.vector);
	}
      else if (axis == 1)
	{
	  gsl_vector_complex_const_view view
	    = gsl_matrix_complex_const_column(tmp, (gsl_permutation_data(p))[i]);
	  gsl_matrix_complex_set_col(m, i, &view.vector);
	}
    }

  return GSL_SUCCESS;
}


/** 
 * Get the sum of the elements of a matrix over each row.
 * \param[out] sum Vector of the sum of the rows.
 * \param[in]  m   Matrix over which to sum.
 * \return         Exit status.
 */
int
gsl_matrix_get_rowsum(gsl_vector *sum, const gsl_matrix *m)
{
  size_t i;
  
  for (i = 0; i < m->size1; i++)
    {
      gsl_vector_const_view view = gsl_matrix_const_row(m, i);
      gsl_vector_set(sum, i, gsl_vector_get_sum(&view.vector));
    }

  return 0;
}
  

/** 
 * Get the sum of the elements of a matrix over each column.
 * \param[out] sum Vector of the sum of the columns.
 * \param[in]  m   Matrix over which to sum.
 * \return         Exit status.
 */
int
gsl_matrix_get_colsum(gsl_vector *sum, const gsl_matrix *m)
{
  size_t i;
  
  for (i = 0; i < m->size2; i++)
    {
      gsl_vector_const_view view = gsl_matrix_const_column(m, i);
      gsl_vector_set(sum, i, gsl_vector_get_sum(&view.vector));
    }

  return 0;
}
  

/** 
 * Get the mean of the elements of a matrix along a given axis.
 * \param[out] mean Vector of the means.
 * \param[in]  m    Matrix over which to calculate the mean.
 * \param[in]  axis Axis over which to calculate the mean
 *                  (0: along columns, 1: along rows).
 * \return          Exit status.
 */
int
gsl_matrix_get_mean(gsl_vector *mean, const gsl_matrix *m, const size_t axis)
{
  switch (axis)
    {
    case 0:
      gsl_matrix_get_colsum(mean, m);
      gsl_vector_scale(mean, 1. / m->size1);
      break;
    case 1:
      gsl_matrix_get_rowsum(mean, m);
      gsl_vector_scale(mean, 1. / m->size2);
      break;
    default:
      GSL_ERROR("axis should be 0 or 1.", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}
  

/** 
 * Get the variance of the elements of a matrix along a given axis.
 * \param[out] var Vector of the variances.
 * \param[in] m    Matrix over which to calculate the variance.
 * \param[in] axis Axis over which to calculate the variance
 *                 (0: along columns, 1: along rows).
 * \return         Exit status.
 */
int
gsl_matrix_get_var(gsl_vector *var, const gsl_matrix *m, const size_t axis)
{
  gsl_vector *mean;
  gsl_matrix *m2;
  size_t sizeAxis;

  switch (axis)
    {
    case 0:
      sizeAxis = m->size2;
      break;
    case 1:
      sizeAxis = m->size1;
      break;
    default:
      GSL_ERROR("axis should be 0 or 1.", GSL_EINVAL);
    }

  if (var->size != sizeAxis)
    {
      GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		 GSL_EINVAL);
    }
      
  //! Allocate
  mean = gsl_vector_alloc(sizeAxis);
  m2 = gsl_matrix_alloc(m->size1, m->size2);

  // Get matrix of squared elements
  gsl_matrix_memcpy(m2, m);
  gsl_matrix_mul_elements(m2, m);

  // Get mean of squares
  gsl_matrix_get_mean(var, m2, axis);

  // remove mean
  gsl_matrix_get_mean(mean, m, axis);
  gsl_vector_mul(mean, mean);
  gsl_vector_sub(var, mean);
  
  // Free
  gsl_matrix_free(m2);
  gsl_vector_free(mean);
  
  return GSL_SUCCESS;
}
  
/** 
 * Get the standard deviation of the elements of a matrix along a given axis.
 * \param[out] std  Vector of the standard deviations.
 * \param[in]  m    Matrix over which to calculate the standard deviation.
 * \param[in]  axis Axis over which to calculate the standard deviation
 *                  (0: along columns, 1: along rows).
 * \return          Exit status.
 */
int
gsl_matrix_get_std(gsl_vector *std, const gsl_matrix *m, const size_t axis)
{
  size_t sizeAxis;

  switch (axis)
    {
    case 0:
      sizeAxis = m->size2;
      break;
    case 1:
      sizeAxis = m->size1;
      break;
    default:
      GSL_ERROR("axis should be 0 or 1.", GSL_EINVAL);
    }

  if (std->size != sizeAxis)
    {
      GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		GSL_EINVAL);
    }
      
  // Get variance
  gsl_matrix_get_var(std, m, axis);

  // Get square root
  gsl_vector_sqrt(std);
  
  return GSL_SUCCESS;
}
  
/** 
 * Get the min of the elements of a matrix along a given axis.
 * \param[out] min  Vector of the min.
 * \param[in]  m    Matrix over which to calculate the min.
 * \param[in]  axis Axis over which to calculate the min
 *                  (0: along columns, 1: along rows).
 * \return          Exit status.
 */
int
gsl_matrix_get_min(gsl_vector *min, const gsl_matrix *m, const size_t axis)
{
  size_t i;
  
  switch (axis)
    {
    case 0:
      if (min->size != m->size2)
	{
	  GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		    GSL_EINVAL);
	}
      
      for (i = 0; i < m->size2; i++)
	{
	  gsl_vector_const_view view = gsl_matrix_const_column(m, i);
	  gsl_vector_set(min, i, gsl_vector_min(&view.vector));
	}
      break;
    case 1:
      if (min->size != m->size1)
	{
	  GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		    GSL_EINVAL);
	}
      
      for (i = 0; i < m->size1; i++)
	{
	  gsl_vector_const_view view = gsl_matrix_const_row(m, i);
	  gsl_vector_set(min, i, gsl_vector_min(&view.vector));
	}
      break;
    default:
      GSL_ERROR("axis should be 0 or 1.", GSL_EINVAL);
    }
  
  return GSL_SUCCESS;
}
  
/** 
 * Get the max of the elements of a matrix along a given axis.
 * \param[out] max  Vector of the max.
 * \param[in]  m    Matrix over which to calculate the max.
 * \param[in]  axis Axis over which to calculate the max
 *                  (0: along columns, 1: along rows).
 * \return          Exit status.
 */
int
gsl_matrix_get_max(gsl_vector *max, const gsl_matrix *m, const size_t axis)
{
  size_t i;
  
  switch (axis)
    {
    case 0:
      if (max->size != m->size2)
	{
	  GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		    GSL_EINVAL);
	}
      
      for (i = 0; i < m->size2; i++)
	{
	  gsl_vector_const_view view = gsl_matrix_const_column(m, i);
	  gsl_vector_set(max, i, gsl_vector_max(&view.vector));
	}
      break;
    case 1:
      if (max->size != m->size1)
	{
	  GSL_ERROR("Output vector should have the same size as\
the number of columns (axis = 0) or rows (axis = 1) of the matrix",
		    GSL_EINVAL);
	}
      
      for (i = 0; i < m->size1; i++)
	{
	  gsl_vector_const_view view = gsl_matrix_const_row(m, i);
	  gsl_vector_set(max, i, gsl_vector_max(&view.vector));
	}
      break;
    default:
      GSL_ERROR("axis should be 0 or 1.", GSL_EINVAL);
    }
  
  return GSL_SUCCESS;
}
  
/** 
 * Get the sum of the elements of a compressed matrix over each row.
 * \param[out] sum Resulting vector of the sum of the rows.
 * \param[in]  m   Compressed matrix over which to sum.
 * \return         Exit status.
 */
int
gsl_spmatrix_get_rowsum(gsl_vector *sum, const gsl_spmatrix *m)
{
  size_t n, outerIdx, p;
    
  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  sum->data[m->i[n] * sum->stride] += m->data[n];
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->size2; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[m->i[p] * sum->stride] += m->data[p];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->size1; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[outerIdx * sum->stride] += m->data[p];
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}

/** 
 * Get the sum of the elements of a compressed matrix over each column.
 * \param[out] sum Resulting vector of the sum of the columns.
 * \param[in]  m   Compressed matrix over which to sum.
 * \return         Exit status.
 */
int
gsl_spmatrix_get_colsum(gsl_vector *sum, const gsl_spmatrix *m)
{
  size_t outerIdx, p, n;
    
  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  sum->data[m->p[n] * sum->stride] += m->data[n];
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->size2; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[outerIdx * sum->stride] += m->data[p];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->size1; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[m->i[p] * sum->stride] += m->data[p];
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }
  
  return GSL_SUCCESS;
}

/** 
 * Get the sum of the elements of a sparse matrix.
 * \param[in] m    Sparse matrix over which to sum.
 * \return         Sum over all the elements of the sparse matrix.
 */
double
gsl_spmatrix_get_sum(const gsl_spmatrix *m)
{
  double sum = 0.;
  size_t n;

  for (n = 0; n < m->nz; n++)
    {
      sum += m->data[n];
    }

  return sum;
}

/** 
 * Divide each row of a compressed matrix by the corresponding vector element.
 * \param[in] m    compressed matrix to divide.
 * \param[in] v    vector to divide the rows of the sparse matrix.
 * \param[in] tol  threshold under which to avoid division by zero.
 * \return         Exit status.
 */
int
gsl_spmatrix_div_rows(gsl_spmatrix *m, const gsl_vector *v, const double tol)
{
  size_t outerIdx, p, n;
  const double tol2 = gsl_pow_2(tol);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  if (gsl_pow_2(v->data[m->i[n] * v->stride]) > tol2)
	    {
	      m->data[n] /= v->data[m->i[n] * v->stride];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->size2; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[m->i[p] * v->stride]) > tol2)
		{
		  m->data[p] /= v->data[m->i[p] * v->stride];
		}
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->size1; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[outerIdx * v->stride]) > tol2)
		{
		  m->data[p] /= v->data[outerIdx * v->stride];
		}
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}

/** 
 * Divide each column of a compressed matrix by the corresponding vector element.
 * \param[in] m    compressed matrix to divide.
 * \param[in] v    vector to divide the columns of the sparse matrix.
 * \param[in] tol  threshold under which to avoid division by zero.
 * \return         Exit status.
 */
int
gsl_spmatrix_div_cols(gsl_spmatrix *m, const gsl_vector *v, const double tol)
{
  size_t outerIdx, p, n;
  const double tol2 = gsl_pow_2(tol);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  if (gsl_pow_2(v->data[m->p[n] * v->stride]) > tol2)
	    {
	      m->data[n] = m->data[n] / v->data[m->p[n] * v->stride];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->size2; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[outerIdx * v->stride]) > tol2)
		{
		  m->data[p] /= v->data[outerIdx * v->stride];
		}
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->size1; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[m->i[p] * v->stride]) > tol2)
		{
		  m->data[p] /= v->data[m->i[p] * v->stride];
		}
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}


/** 
 * Pre-allocate a sparse matrix in triplet format to read from a binary stream.
 * \param[in] stream Stream from which the sparse matrix will be read.
 * \return    m      The allocated sparse matrix.
 */
gsl_spmatrix *
gsl_spmatrix_alloc2read(FILE *stream, const size_t type)
{
  gsl_spmatrix *m;
  size_t size1, size2, nz;
  size_t items;

  // 
  items = fread(&size1, sizeof(size_t), 1, stream);
  if (items != 1)
    {
      GSL_ERROR_NULL("fread failed on size1", GSL_EFAILED);
    }

  items = fread(&size2, sizeof(size_t), 1, stream);
  if (items != 1)
    {
      GSL_ERROR_NULL("fread failed on size2", GSL_EFAILED);
    }

  items = fread(&nz, sizeof(size_t), 1, stream);
  if (items != 1)
    {
      GSL_ERROR_NULL("fread failed on nz", GSL_EFAILED);
    }

  // Allocate
  m = gsl_spmatrix_alloc_nzmax(size1, size2, nz, type);

  if (!m)
    {
      GSL_ERROR_NULL ("error allocating m", GSL_ENOMEM);
    }

  // Rewind the stream for the true reading
  rewind(stream);

  return m;
}
