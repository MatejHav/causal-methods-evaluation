from builders import Experiment

def main():
    dimensions = 50
    Experiment().add_causal_forest()\
        .add_causal_forest(honest=False)\
        .add_mean_squared_error()\
        .add_absolute_error()\
        .add_no_noise_generator(dimensions)\
        .add_large_noise_generator(dimensions)\
        .add_treatment_and_no_treatment_have_same_effect_generator(dimensions)\
        .add_biased_generator(dimensions)\
        .run(save_data=True, save_graphs=True, show_graphs=False)

if __name__ == '__main__':
    main()