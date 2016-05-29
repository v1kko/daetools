TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = LA_SuperLU \
    LA_Trilinos_Amesos \
    LA_IntelPardiso \
    LA_Pardiso \
    pySuperLU \
    pyTrilinos \
    pyIntelPardiso \
    pyPardiso \
    FE_DealII \
    NLOPT_NLPSolver \
    BONMIN_MINLPSolver \
    pyNLOPT \
    pyBONMIN \
    pyDealII
