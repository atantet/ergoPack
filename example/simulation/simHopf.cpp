#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <SDESolvers.hpp>
#include <SDDESolvers.hpp>
#include <ODEFields.hpp>
#include <gsl_extension.hpp>
#include <omp.h>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simSDDE.cpp 
 *  \ingroup examples
 *  \brief Simulate a Stochastic Delay Differential Equation.
 *
 *  Simulate a Stochastic Delay Differential Equation
 *  with polynomial vector fields.
 */

bool isStationaryPoint(const gsl_matrix *X, const double gap,
		       const double printStep, const double tol);


/** \brief Simulation of an Stochastic Delay Differential Equation.
 *
 *  Simulation of an Stochastic Delay Differential Equation.
 *  After parsing the configuration file,
 *  a delayed polynomial vector field for the drift, a diffusion matrix
 *  and an Euler-Maruyama stochastic numerical scheme are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  char dstPostfix[256], srcPostfix[256];

  // Read configuration file
  if (argc < 2)
    {
      std::cout << "Enter path to configuration file:" << std::endl;
      std::cin >> configFileName;
    }
  else
    {
      strcpy(configFileName, argv[1]);
    }
  try
    {
      Config cfg;
      std::cout << "Sparsing config file " << configFileName << std::endl;
      cfg.readFile(configFileName);
      readGeneral(&cfg);
      readModel(&cfg);
      readSimulation(&cfg);
      readSprinkle(&cfg);
      std::cout << "Sparsing success.\n" << std::endl;
    }
  catch(const SettingTypeException &ex) {
    std::cerr << "Setting " << ex.getPath() << " type exception."
	      << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "Setting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "Setting " << ex.getPath() << " name exception."
	      << std::endl;
    throw ex;
  }
  catch(const ParseException &ex) {
    std::cerr << "Parse error at " << ex.getFile() << ":" << ex.getLine()
              << " - " << ex.getError() << std::endl;
    throw ex;
  }
  catch(const FileIOException &ex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw ex;
  }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(dstPostfix, "%s_mu%04d_alpha%04d_gamma%04d_delta%04d_beta%04d_eps%04d_L%d_spinup%d_dt%d_samp%d", srcPostfix,
	  (int) (p["mu"] * 10000 + 0.1), (int) (p["alpha"] * 10000 + 0.1),
	  (int) (p["gamma"] * 10000 + 0.1), (int) (p["delta"] * 10000 + 0.1),
	  (int) (p["beta"] * 10000 + 0.1), (int) (p["eps"] * 10000 + 0.1),
	  (int) L, (int) spinup, (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);

#pragma omp parallel
  {
    FILE *dstStream;
    gsl_vector *init = gsl_vector_alloc(dim);
    gsl_matrix *X;
    size_t seed;
    gsl_rng * r;
    char dstFileName[256];

    // Configure random number generator
    gsl_rng_env_setup(); // Get GSL_RNG_TYPE [=gsl_rng_mt19937] and GSL_RNG_SEED [=0]
    r = gsl_rng_alloc(gsl_rng_ranlxs1);
    // Get seed and set random number generator
    seed = (size_t) (1 + omp_get_thread_num());
#pragma omp critical
    {
      std::cout << "Setting random number generator with seed: " << seed
		<< std::endl;
      std::cout.flush();
    }
    gsl_rng_set(r, seed);

    // Define field
    vectorField *field = new Hopf(&p);

    // Define stochastic vector field
    vectorFieldStochastic *stocField = new additiveWiener(Q, r);

    // Define numerical scheme
    numericalSchemeStochastic *scheme = new EulerMaruyama(dim);

    // Define model
    modelStochastic *mod = new modelStochastic(field, stocField, scheme);
  
    // Set initial state
#pragma omp for
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      bool stat = true;
      while (stat) {
	// Get random initial distribution
	for (size_t d = 0; d < (size_t) dim; d++)
	  gsl_vector_set(init, d,
			 gsl_ran_flat(r, gsl_vector_get(minInitState, d),
				      gsl_vector_get(maxInitState, d)));

	// Set initial state
#pragma omp critical
	{
	  printf("Setting initial state to (%.1lf, %.1lf, %.1lf)\n",
		 gsl_vector_get(init, 0),
		 gsl_vector_get(init, 1),
		 gsl_vector_get(init, 2));
	}
	mod->setCurrentState(init);

	// Numerical integration of spinup
#pragma omp critical
	{
	  std::cout << "Integrating spinup " << traj << std::endl;
	}
	X = gsl_matrix_alloc(1, 1); // Fake allocation
	mod->integrate(init, spinup, dt, 0., printStepNum, &X);

	// Check if stationary point
	if (isStationaryPoint(X, 0.1, printStep, 1.e-8)) {
#pragma omp critical
	  {
	    std::cout << "Trajectory converged to point. Continue..."
		      << std::endl;
	  }
	}
	else
	  stat = false;
      }

      // Numerical integration
#pragma omp critical
      {
	std::cout << "Integrating trajectory " << traj << std::endl;
      }
      mod->integrate(LCut, dt, 0., printStepNum, &X);

      // Write results
#pragma omp critical
      {
	sprintf(dstFileName, "%s/simulation/sim%s_traj%d.%s",
		resDir, dstPostfix, (int) traj, fileFormat);
	if (!(dstStream = fopen(dstFileName, "w"))) {
	  std::cerr << "Can't open " << dstFileName
		    << " for writing simulation: " << std::endl;;
	  perror("");
	}

	std::cout << "Writing " << traj << std::endl;
	if (strcmp(fileFormat, "bin") == 0)
	  gsl_matrix_fwrite(dstStream, X);
	else
	  gsl_matrix_fprintf(dstStream, X, "%f");
	fclose(dstStream);  
      }

      // Free
      gsl_matrix_free(X);
    }
    delete stocField;
    delete scheme;
    delete mod;
    gsl_rng_free(r);
  }
  freeConfig();
  
  return 0;
}

bool
isStationaryPoint(const gsl_matrix *X, const double gap,
		  const double printStep, const double tol) {
  const size_t nGap = (size_t) (gap / printStep + 0.1);
  double dist;
  gsl_vector *vec = gsl_vector_alloc(X->size2);
  gsl_vector_const_view vView
    = gsl_matrix_const_row(X, X->size1 - nGap - 1);

  gsl_matrix_get_row(vec, X, X->size1 - 1);
  gsl_vector_sub(vec, &vView.vector);
  dist = gsl_vector_get_norm(vec);
  gsl_vector_free(vec);

  return (dist < tol);
}
