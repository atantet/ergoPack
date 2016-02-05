Main page                    {#mainpage}
=========

Introduction                  {#introduction}
============

ergoPack is a collection of scientific routines in C++
originally developed by Alexis Tantet for _research purpose_.
These codes are open source in order to promote reproducibility.
Visit Alexis' [home page][UU] for contact.

[TOC]

Installation               {#installation}
============

Getting the code                {#getting-the-code}
----------------

First the ergoPack repository should be cloned using [git].
To do so:
1. Change directory to where you want to clone the repository $GITDIR:

        cd $GITDIR
     
2. Clone the ergoPack repository (git should be installed):

        git clone https://github.com/atantet/ergoPack
     
Dependencies          {#dependencies}
------------

Mandatory libraries:
- [GSL] is used as the main C scientific library.
- [Eigen] is a C++ template library for linear algebra, mainly used for sparse matrices manipulation.

Specific libraries:
- [ARPACK++] is an object-oriented version of the ARPACK package used to calculate the spectrum of sparse matrices in atspectrum.hpp.
- [OpenMP][OMP] is used for multi-threading by transferOperator.hpp
when WITH_OMP is set to 1 when compiling.

Installing the code                {#installing-the-code}
-------------------

1. Create a directory ergoPack/ in your favorite include directory $INCLUDE:

        mkdir $INCLUDE/ergoPack
     
2. Copy the ergoPack/*.hpp source files to $INCLUDE/ergoPack/:

        cd $GITDIR/ergoPack
        cp *.hpp $INCLUDE/ergoPack
     
3. Include these files in your C++ codes. For example, in order to include the matrix manipulation functions in atmatrix.hpp,
add in your C++ file:

        #include <ergoPack/atmatrix.hpp>
    

Updating the code              {#updating-the-code}
-----------------

1. Pull the ergoPack repository:

        cd $GITDIR/ergoPack     
        git pull
     
2. Copy the source files to your favorite include directory $INCLUDE:

        cp *.hpp $INCLUDE/ergoPack


Compiling                   {#compiling}
=========

Without OpenMP               {#without-omp}
--------------

If INCLUDE is not a system directory such as /usr/include/ or /usr/local/include/
then either it should be added to CPLUS_INCLUDE_PATH or at compilation using -I$INCLUDE. E.g.

     g++ -c -I$INCLUDE source.cpp

When linking, GSL should be linked by added -lgsl.
If GSL's directory is not a system one or in LIBRARY_PATH then -L$GSLDIR should be added. E.g.

     g++ -L$GSLDIR source.o -lgsl
     
With OpenMP                  {#with-omp}
-----------

If OpenMP is to be used, then WITH_OMP should be set to 1,
-fopenmp -DWITH_OMP=$WITH_OMP used when compiling
and -lgomp when linking.

     g++ -c -fopenmp -DWITH_OMP=$WITH_OMP -I$INCLUDE source.cpp
     g++ -L$GSLDIR source.o -lgsl -lgomp

Disclaimer                  {#disclaimer}
==========

These codes are developed for _research purpose_.
_No warranty_ is given regarding their robustess.

[UU]: http://www.uu.nl/staff/AJJTantet/ "Alexis' personal page"
[git]: https://git-scm.com/ "git"
[ergoPack_doc]: http://atantet.github.io/ergoPack/ "ergoPack documentation"
[GSL]: http://www.gnu.org/software/gsl/ "GSL - GNU Scientific Library"
[Eigen]: http://eigen.tuxfamily.org/ "Eigen"
[ARPACK++]: http://www.caam.rice.edu/software/ARPACK/arpack++.html "ARPACK++"
[OMP]: http://www.openmp.org/ "OpenMP"
