"""
Coherent Entropy Reactor (CER)
A recursive network that weighs its own mind.
"""

__version__ = "0.1.0"
__author__ = "Anthony J Vasquez Sr"

from .core.reactor import CoherentEntropyReactor
from .entropy.measurement import SemanticMass, CommutationCost
from .liquid.dynamics import LiquidLayer, KuramotoOscillator
