from builders import Experiment

def main():
    dimensions = 50
    Experiment().add_causal_forest()\
        .add_mean_squared_error()\
        .add_absolute_error()\
        .add_treatment_effect_offset_generator(dimensions=dimensions)\
        .run(save_data=True, save_graphs=True, show_graphs=False)

if __name__ == '__main__':
    main()