from imitation.algorithms import preference_comparisons
from sacred import Ingredient

fragmenter_ingredient = Ingredient("fragmenter")


@fragmenter_ingredient.capture
def create_fragmenter(custom_logger, rng, kwargs=None):
    return preference_comparisons.RandomFragmenter(
        **kwargs if kwargs else {},
        rng=rng,
        custom_logger=custom_logger,
    )