[![Language: Python](https://img.shields.io/badge/language-Python_(3.8.0%2B)-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP 8, 257](https://img.shields.io/badge/code%20style-PEP%208%2C%20257-orange.svg)](https://www.python.org/dev/peps)
[![Build Status](https://travis-ci.com/THargreaves/mytrix.svg?branch=master)](https://travis-ci.com/THargreaves/mytrix)

# Mytrix

_A Python package for handling matrices and other linear algebra concepts_

## About

Mytrix is a collaborative project that I began in the summer of 2019. This is not a package of much practical use but was rather developed as a way to practice Python package development and collaborative coding. The scope of the package is reasonably wide and loosely-defined; essentially, anything that is vaguely related to linear algebra is welcome for inclusion.

The philosophy of the package is heavily inspired by MATLAB's use of multiple dispatch and operator overloading to enable efficiency savings when a given input statisfies certain mathematical properties. Further, the core contributors for this project are all mathematicians or statisticians and so the syntax and functionality of the package is most relevant to such an audience.

![Multiple dispatch in MATLAB for solving systems of linear equations (this is flowchart is for dense matrices alone; sparse matrices have their own, even larger flowchart)](https://uk.mathworks.com/help/matlab/ref/mldivide_full.png)

Although this package is in no way a replacement for any of the existing linear algebra solutions ([NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) for example) due to its comparatively small scope, inefficient design (no C code here), and lack of robustness, I hope that people will still be interested in contributing the project to help their own learning and others can benefit from looking at the design of a linear algebra code base that isn't as verbose or obfuscated as the well-known solutions.

## Contributing

The core contributors to this project are:

* Tim Hargreaves [[GitHub](https://github.com/THargreaves) | [LinkedIn](https://www.linkedin.com/in/tim-hargreaves/)]

Anyone is free (and encouraged) to contribute to this code base. All I ask is that one follows these key rules:

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257/). If you struggle to remember these conventions, please install a relevant linter
* Document and write tests for any new features that you implement
* Attempt to follow existing naming conventions, especially for function signatures and temporary variables