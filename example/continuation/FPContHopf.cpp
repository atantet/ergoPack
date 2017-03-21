#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"


/** \file FPCont.cpp 
 *  \brief Fixed point continuation in the Hopf normal form.
 *
 *  Fixed point continuation in the Hopf normal form.
 */


/** \brief Fixed point continuation in the Hopf normal form.
 *
 *  Fixed point continuation in the Hopf normal form.
 */
int main(int argc, char * argv[])
{
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
     readContinuation(&cfg);
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

  gsl_matrix *solJac = gsl_matrix_alloc(dim + 1, dim + 1);
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *eigVec = gsl_matrix_complex_alloc(dim, dim);
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(dim);
  gsl_matrix_view jac;
  char dstFileName[256], dstFileNameEigVec[256], dstFileNameEigVal[256],
    srcPostfix[256], dstPostfix[256], contPostfix[256];
  FILE *dstStream, *dstStreamEigVec, *dstStreamEigVal;


  // Define names and open destination file
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(contPostfix, "_cont%04d_contStep%de%d", 0,
	  (int) (mantis*1.01), (int) (exp*1.01));
  sprintf(dstPostfix, "%s_beta%04d_gamma%04d%s", srcPostfix,
	  (int) (p["beta"] * 1000 + 0.1), (int) (p["gamma"] * 1000 + 0.1),
	  contPostfix);
  sprintf(dstFileName, "%s/continuation/fpState/fpState%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
   {
     fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  sprintf(dstFileNameEigVec, "%s/continuation/fpEigVec/fpEigVecCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamEigVec = fopen(dstFileNameEigVec, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(dstFileNameEigVal, "%s/continuation/fpEigVal/fpEigValCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamEigVal = fopen(dstFileNameEigVal, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }

  // Start from fixed point for mu = 0 (x = 0)
  gsl_vector_free(initCont);
  initCont = gsl_vector_alloc(dim + 1);
  gsl_vector_set(initCont, 0, 0.);
  gsl_vector_set(initCont, 1, 0.);
  gsl_vector_set(initCont, 2, -5.);
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new HopfCont(&p);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianHopfCont(&p);

  // Define fixed point problem
  fixedPointCont *track = new fixedPointCont(field, Jacobian, epsDist,
					     epsStepCorrSize, maxIter,
					     verbose);

  // First correct
  if (verbose)
    {
      std::cout << "Initial state: " << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
    }
  std::cout << "Applying initial correction..." << std::endl;
  track->correct(initCont);

  if (!track->hasConverged())
    {
      std::cerr << "First correction could not converge." << std::endl;
      return -1;
    }
  else
    std::cout << "Found initial fixed point after "
	      << track->getNumIter() << " iterations"
	      << " with distance = " << track->getDist()
	      << " and step = " << track->getStepCorrSize() << std::endl;


  while ((gsl_vector_get(initCont, dim) >= contMin)
	 && (gsl_vector_get(initCont, dim) <= contMax))
    {
      // Find fixed point
      std::cout << "\nApplying continuation step..." << std::endl;
      track->continueStep(contStep);

      if (!track->hasConverged())
	{
	  std::cerr << "Continuation could not converge." << std::endl;
	  break;
	}
      else
	std::cout << "Found initial fixed point after "
		  << track->getNumIter() << " iterations"
		  << " with distance = " << track->getDist()
		  << " and step = " << track->getStepCorrSize() << std::endl;

      // Get solution and the Jacobian
      track->getCurrentState(initCont);
      track->getStabilityMatrix(solJac);
      jac = gsl_matrix_submatrix(solJac, 0, 0, dim, dim);

      // Find eigenvalues
      gsl_eigen_nonsymmv(&jac.matrix, eigVal, eigVec, w);

      // Print fixed point
      std::cout << "Fixed point:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      std::cout << "Eigenvalues:" << std::endl;
      gsl_vector_complex_fprintf(stdout, eigVal, "%lf");

      // Write results
      if (strcmp(fileFormat, "bin") == 0)
	{
	  gsl_vector_fwrite(dstStream, initCont);
	  gsl_vector_complex_fwrite(dstStreamEigVal, eigVal);
	  gsl_matrix_complex_fwrite(dstStreamEigVec, eigVec);
	}
      else
	{
	  gsl_vector_fprintf(dstStream, initCont, "%lf");
	  gsl_vector_complex_fprintf(dstStreamEigVal, eigVal, "%lf");
	  gsl_matrix_complex_fprintf(dstStreamEigVec, eigVec, "%lf");
	}
      
      // Flush in case premature exit
      fflush(dstStream);
      fflush(dstStreamEigVal);
      fflush(dstStreamEigVec);
    }
  
  gsl_eigen_nonsymmv_free(w);
  gsl_vector_complex_free(eigVal);
  gsl_matrix_complex_free(eigVec);
  delete track;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solJac);
  fclose(dstStreamEigVal);
  fclose(dstStreamEigVec);
  fclose(dstStream);  
  freeConfig();

  return 0;
}
