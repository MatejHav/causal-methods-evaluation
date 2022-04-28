from builders import Experiment

def main():
    dimensions = 5
    Experiment().add_causal_forest()\
        .add_dragonnet(dimensions=dimensions)\
        .add_mean_squared_error()\
        .add_all_effects_generator(dimensions)\
        .add_only_treatment_effect_generator(dimensions)\
        .add_no_treatment_effect_generator(dimensions)\
        .run(save_data=True, save_graphs=True, show_graphs=False)

if __name__ == '__main__':
    main()