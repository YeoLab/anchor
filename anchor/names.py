"""
Names of the modalities
"""

# Set constants of the names of the models so they can always be referenced
# as variables rather than strings

# Most of the density is at 0
NEAR_ZERO = 'excluded'

# Old "middle" modality - most of the density is at 0.5
NEAR_HALF = 'middle'

# Most of the density is at 1
NEAR_ONE = 'included'

# The density is split between 0 and 1
BIMODAL = 'bimodal'

# Cannot decide on one of the above models (the null model fits better) so use
# this model instead
NULL_MODEL = 'uncategorized'
