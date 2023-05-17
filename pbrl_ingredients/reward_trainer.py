from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import EnsembleTrainer, BasicRewardTrainer
from sacred import Ingredient

reward_trainer_ingredient = Ingredient("reward_trainer")


@reward_trainer_ingredient.capture
def create_reward_trainer(preference_model, rng, kwargs=None):

    loss = preference_comparisons.CrossEntropyRewardLoss()

    if preference_model.ensemble_model is not None:
        return EnsembleTrainer(
            preference_model,
            loss=loss,
            rng=rng,
            **kwargs if kwargs else {},
        )
    else:
        return BasicRewardTrainer(
            preference_model,
            loss=loss,
            rng=rng,
            **kwargs if kwargs else {},
        )