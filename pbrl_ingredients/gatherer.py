from imitation.algorithms.preference_comparisons import SyntheticGatherer
from sacred import Ingredient

GATHERERS = {"synthetic": SyntheticGatherer}

gatherer_ingredient = Ingredient("gatherer")


@gatherer_ingredient.config
def cfg():
    gatherer_type = "synthetic"


@gatherer_ingredient.capture
def create_gatherer(rng):
    return SyntheticGatherer(rng=rng)
