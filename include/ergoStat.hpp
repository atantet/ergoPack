#ifndef ERGOSTAT_HPP
#define ERGOSTAT_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl_extension.hpp>


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

  
#endif
