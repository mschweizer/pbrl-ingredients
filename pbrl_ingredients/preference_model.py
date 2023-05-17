from imitation.algorithms import preference_comparisons
from sacred import Ingredient

preference_model_ingredient = Ingredient("preference_model")


@preference_model_ingredient.capture
def create_preference_model(reward_net, kwargs=None):
    return preference_comparisons.PreferenceModel(
        **kwargs if kwargs else {},
        model=reward_net,
    )
