import warnings

# Supress annoying RuntimeWarnings that yes_no_t and colperm_t 
# to-Python converters are already registered
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import pySuperLU_MT
