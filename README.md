Introduction
============

ATSuite C++ is a collection of scientific routines in C++
originally developed by Alexis Tantet for _research purpose_.
These codes are open source in order to promote reproducibility.
Visit Alexis' [home page][UU] for contact.
The full documentation can be found at [ATSuite C++][ATSuite_cpp_doc].


Table of contents
=================

  * [Installation] (#installation)
    + [Getting the code] (#getting-the-code)
    + [Dependencies] (#dependencies)
    + [Installing the code] (#installing-the-code)
    + [Updating the code] (#updating-the-code)
  * [Compiling] (#compiling)
    + [Without OpenMP] (#without-openmp)
    + [With OpenMP] (#with-openmp)
  * [Disclaimer] (#disclaimer)
  

Installation
============

Getting the code
----------------

First the ATSuite_cpp repository should be cloned using [git].
To do so:
1. Change directory to where you want to clone the repository $GITDIR:

        cd $GITDIR
     
2. Clone the ATSuite_cpp repository (git should be installed):

        git clone https://github.com/atantet/ATSuite_cpp
     
Dependencies
------------

Mandatory libraries:
- [GSL] is used as the main C scientific library.
- [Eigen] is a C++ template library for linear algebra, mainly used for sparse matrices manipulation.

Specific libraries:
- [ARPACK++] is an object-oriented version of the ARPACK package used to calculate the spectrum of sparse matrices in atspectrum.hpp.
- [OpenMP][OMP] is used for multi-threading by transferOperator.hpp
when WITH_OMP is set to 1 when compiling.

Installing the code
-------------------

1. Create a directory ATSuite/ in your favorite include directory $INCLUDE:

        mkdir $INCLUDE/ATSuite
     
2. Copy the ATSuite_cpp/*.hpp source files to $INCLUDE/ATSuite/:

        cd $GITDIR/ATSuite_cpp
        cp *.hpp $INCLUDE/ATSuite
     
3. Include these files in your C++ codes. For example, in order to include the matrix manipulation functions in atmatrix.hpp,
add in your C++ file:

        #include <ATSuite/atmatrix.hpp>
    

Updating the code
-----------------

1. Pull the ATSuite_cpp repository:

        cd $GITDIR/ATSuite_cpp     
        git pull
     
2. Copy the source files to your favorite include directory $INCLUDE:

        cp *.hpp $INCLUDE/ATSuite


Compiling
=========

Without OpenMP
--------------

If INCLUDE is not a system directory such as /usr/include/ or /usr/local/include/
then either it should be added to CPLUS_INCLUDE_PATH or at compilation using -I$INCLUDE. E.g.

     g++ -c -I$INCLUDE source.cpp

When linking, GSL should be linked by added -lgsl.
If GSL's directory is not a system one or in LIBRARY_PATH then -L$GSLDIR should be added. E.g.

     g++ -L$GSLDIR source.o -lgsl
     
With OpenMP
-----------

If OpenMP is to be used, then WITH_OMP should be set to 1,
-fopenmp -DWITH_OMP=$WITH_OMP used when compiling
and -lgomp when linking.

     g++ -c -fopenmp -DWITH_OMP=$WITH_OMP -I$INCLUDE source.cpp
     g++ -L$GSLDIR source.o -lgsl -lgomp

Disclaimer
==========

These codes are developed for _research purpose_.
_No warranty_ is given regarding their robustess.

[UU]: http://www.uu.nl/staff/AJJTantet/ "Alexis' personal page"
[git]: https://git-scm.com/ "git"
[ATSuite_cpp_doc]: http://atantet.github.io/ATSuite_cpp/ "ATSuite C++ documentation"
[GSL]: http://www.gnu.org/software/gsl/ "GSL - GNU Scientific Library"
[Eigen]: http://eigen.tuxfamily.org/ "Eigen"
[ARPACK++]: http://www.caam.rice.edu/software/ARPACK/arpack++.html "ARPACK++"
[OMP]: http://www.openmp.org/ "OpenMP"
