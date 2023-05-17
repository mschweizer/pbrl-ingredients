from imitation.algorithms import preference_comparisons
from imitation.scripts.common.common import common_ingredient
from sacred import Ingredient

pbrl_agent_ingredient = Ingredient("pbrl_agent",
                                   ingredients=[common_ingredient])


@pbrl_agent_ingredient.config
def cfg():
    query_schedule = "hyperbolic"
    num_iterations = 5  # Arbitrary, should be tuned for the task
    fragment_length: int = 25  # timesteps per fragment used for comparisons
    comparison_queue_size = None
    # factor by which to oversample transitions before creating fragments
    trajectory_oversampling_factor = 1
    # fraction of total_comparisons that will be sampled right at the beginning
    initial_comparison_fraction = 0.1
    # fraction of sampled trajectories that will include some random actions
    allow_variable_horizon = True


@pbrl_agent_ingredient.capture
def create_agent(trajectory_generator,
                 fragmenter,
                 gatherer,
                 reward_trainer,
                 reward_net,
                 custom_logger,
                 query_schedule,
                 num_iterations,
                 fragment_length,
                 comparison_queue_size,
                 trajectory_oversampling_factor,
                 initial_comparison_fraction,
                 allow_variable_horizon):

    agent = preference_comparisons.PreferenceComparisons(
        trajectory_generator=trajectory_generator,
        reward_model=reward_net,
        num_iterations=num_iterations,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparison_queue_size=comparison_queue_size,
        fragment_length=fragment_length,
        transition_oversampling=trajectory_oversampling_factor,
        initial_comparison_frac=initial_comparison_fraction,
        custom_logger=custom_logger,
        allow_variable_horizon=allow_variable_horizon,
        query_schedule=query_schedule,
    )

    return agent


