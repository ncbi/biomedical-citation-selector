"""
Threshold variables.

The voting and CNN thresholds
are only used for testing, 
and not in the production model.

Thresholds are chosen in generate_results.py,
by selecting threshold that generates result
closes to .995 recall, for each model. 
The group thresholds are chosen the same way,
but --group_thresh and --group "group" must be provided
"""

# Threshold to label citations as out of scope, at a 99.5% confidence level
COMBINED_THRESH = .0015
# Threshold to label citations as in scope, at a 97% confidence level
PRECISION_THRESH = .7290

# Voting threshold
# Used for testing performance
VOTING_THRESH = .034
VOTING_JURISPRUDENCE_THRESH = .185
VOTING_SCIENCE_THRESH = .075

# CNN thresholds
# Used for testing performance
CNN_THRESH = .0237
CNN_JURISPRUDENCE_THRESH = .45
CNN_SCIENCE_THRESH = .15

# Group thresholds. This is currently not recommeded for use. 
SCIENCE_THRESH = .0036 #.02
JURISPRUDENCE_THRESH = .075 #Halved from .15 
BIOTECH_THRESH = .0007
CHEMISTRY_THRESH = .0017
