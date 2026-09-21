# SALSA — Sectional Aerosol module

**SALSA (Sectional Aerosol module for Large Scale Applications)** is a sectional aerosol microphysics model developed for use in atmospheric chemistry and climate models.

This repository contains the **HAM-SALSA** implementation, including the integration of SALSA with the **ECHAM-HAM** aerosol-climate model framework and its ongoing development for **OpenIFS**.

## Overview

SALSA represents the aerosol size distribution using size sections and explicitly describes the evolution of aerosol particles through processes including:

* nucleation and particle formation
* condensation of trace gases
* coagulation
* aerosol water uptake
* cloud activation
* precipitation and wet removal
* dry deposition
* aerosol transport and mixing

The sectional representation allows aerosol properties and processes to be resolved across a broad range of particle sizes while retaining information on aerosol composition and mixing state.

SALSA has been developed for applications ranging from regional and global atmospheric modelling to aerosol-cloud interaction and climate studies.

## HAM-SALSA

**HAM-SALSA** is the implementation of SALSA within the **HAM** aerosol module framework.

The development combines the sectional SALSA representation with the processes and infrastructure of the HAM aerosol model. HAM-SALSA is intended for applications where an explicit representation of the aerosol size distribution is important, including studies of:

* aerosol-cloud interactions
* cloud condensation nuclei (CCN)
* aerosol indirect effects
* aerosol-radiation interactions
* aerosol lifetime and transport
* new particle formation
* aerosol-cloud-climate feedbacks

The repository is also being developed to provide a SALSA implementation for **OpenIFS**, enabling the use of the sectional aerosol representation within the IFS modelling framework.

## Repository status

This repository is the **current development repository for HAM-SALSA**.

It supersedes the earlier standalone SALSA implementation previously hosted in this repository. The current code should therefore be considered the primary version of SALSA for development within the HAM/OpenIFS framework.

Development is ongoing, and the structure, interfaces and scientific implementation may change as the model is further developed and tested.

## Model structure

The repository contains the SALSA aerosol microphysics together with the interfaces required for coupling it to atmospheric models.

The main components include:

```text
SALSA/
├── aerosol microphysics
├── aerosol processes
├── cloud activation
├── aerosol removal
├── model interfaces
└── OpenIFS / HAM interfaces
```

The exact directory structure is evolving as the OpenIFS implementation is developed.

## Scientific applications

HAM-SALSA can be used to investigate aerosol processes and their effects on the atmosphere and climate, including:

### Aerosol-cloud interactions

SALSA provides an explicit treatment of aerosol size distributions, making it possible to investigate the relationship between aerosol populations, CCN and cloud droplet activation.

### Aerosol-radiation interactions

The size-resolved aerosol representation can be used to calculate aerosol optical properties and investigate direct and semi-direct aerosol effects.

### Aerosol-climate interactions

The model can be used in global climate simulations to study how changes in aerosol emissions affect radiation, clouds and climate.

### New particle formation

The sectional representation allows the evolution of newly formed particles from the smallest size ranges towards larger climatically relevant aerosol sizes to be investigated.

## Model history

SALSA was originally developed as a sectional aerosol module for large-scale atmospheric applications. It has subsequently been implemented and developed within several atmospheric modelling frameworks.

The present repository focuses on the **HAM-SALSA** implementation and its development towards use with **OpenIFS**.

## Development

The model is developed primarily in **Fortran** and is intended for use in large-scale atmospheric modelling systems.

Contributions, bug reports and scientific feedback are welcome.

When reporting a problem, please include, where possible:

* the model version or commit
* compiler and compiler version
* model configuration
* relevant input settings
* a description of the problem
* error messages or diagnostic output

## Licence

See [LICENSE.txt](../LICENSE.txt) for the licence and usage conditions applying to the code in this repository.

## References

The scientific description and development history of SALSA are documented in the relevant SALSA publications.

A reference to the specific HAM-SALSA implementation should be added here as the corresponding implementation paper is published.

## Contact

For questions concerning the SALSA model and its development, please use the GitHub repository issues or contact the SALSA development team.

---

**SALSA — Sectional Aerosol module for Large Scale Applications**

Developed for atmospheric aerosol, cloud and climate modelling.
