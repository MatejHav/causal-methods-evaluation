from builders import Experiment

def main():
    dimensions = 5
    Experiment().add_causal_forest()\
        .add_causal_forest(honest=False)\
        .add_mean_squared_error()\
        .add_absolute_error()\
        .add_all_effects_generator(dimensions, sample_size=2000)\
        .add_no_treatment_effect_generator(dimensions, sample_size=2000) \
        .add_only_treatment_effect_generator(dimensions, sample_size=2000) \
        .add_small_treatment_propensity_generator(dimensions, sample_size=2000)\
        .run(save_data=True, save_graphs=True, show_graphs=False)

if __name__ == '__main__':
    main()