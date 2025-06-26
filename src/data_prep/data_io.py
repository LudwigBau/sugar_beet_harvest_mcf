import os
import pickle
import numpy as np
import pandas as pd



class DataLoader:
    @staticmethod
    def load_synthetic_data(scenario_id, base_file_path):
        # Construct file paths
        distance_matrix_file = os.path.join(base_file_path, f'scenario_{scenario_id}_distance_matrix.npy')
        sugar_concentration_file = os.path.join(base_file_path, f'scenario_{scenario_id}_sugar_concentration.npy')
        c_t_file = os.path.join(base_file_path, f'scenario_{scenario_id}_distance_matrix.npy')
        c_h_file = os.path.join(base_file_path, f'scenario_{scenario_id}_c_h_matrix.npy')
        c_l_file = os.path.join(base_file_path, f'scenario_{scenario_id}_c_l_matrix.npy')
        harvesters_data_file = os.path.join(base_file_path, f'scenario_{scenario_id}_harvesters_data.pkl')
        loaders_data_file = os.path.join(base_file_path, f'scenario_{scenario_id}_loaders_data.pkl')
        beet_yield_file = os.path.join(base_file_path, f'scenario_{scenario_id}_beet_yield.npy')
        field_size_file = os.path.join(base_file_path, f'scenario_{scenario_id}_field_size.npy')
        field_location_file = os.path.join(base_file_path, f'scenario_{scenario_id}_field_location.csv')

        # Load data
        distance_matrix = np.load(distance_matrix_file)
        sugar_concentration = np.load(sugar_concentration_file)
        c_t = np.load(c_t_file)
        c_h = np.load(c_h_file)
        c_l = np.load(c_l_file)
        with open(harvesters_data_file, 'rb') as f:
            harvesters_data = pickle.load(f)
        with open(loaders_data_file, 'rb') as f:
            loaders_data = pickle.load(f)
        beet_yield = np.load(beet_yield_file)
        field_size = np.load(field_size_file)
        field_location = pd.read_csv(field_location_file, index_col=0)

        return {
            "distance_matrix": distance_matrix,
            "sugar_concentration": sugar_concentration,
            "c_t": c_t,
            "c_h": c_h,
            "c_l": c_l,
            "harvesters_data": harvesters_data,
            "loaders_data": loaders_data,
            "beet_yield": beet_yield,
            "field_size": field_size,
            "field_location": field_location
        }
    @staticmethod
    def load_results_data(flow_stage, base_file_path):

        # Define the file paths for each dataset
        file_paths = {
            'field_machine_assignment': os.path.join(base_file_path, f'{flow_stage}_machinery_assignment.pkl'),
            'beet_movement': os.path.join(base_file_path, f'{flow_stage}_beet_movement.pkl'),
            'schedule': os.path.join(base_file_path, f'{flow_stage}_schedule.pkl'),
            'field_yield': os.path.join(base_file_path, f'{flow_stage}_field_yield.pkl'),
            'machinery_cost': os.path.join(base_file_path, f'{flow_stage}_machinery_cost.pkl'),
            'harvester_cost_matrix': os.path.join(base_file_path, f'{flow_stage}_harvester_cost_matrix.pkl'),
            'loader_cost_matrix': os.path.join(base_file_path, f'{flow_stage}_loader_cost_matrix.pkl'),
            'revenue_unmet': os.path.join(base_file_path, f'{flow_stage}_revenue_unmet.pkl')
        }

        # Load all files into a dictionary
        results_dict = {key: pd.read_pickle(path) for key, path in file_paths.items()}

        return results_dict

def save_raw_data(scenario_id, data_generator):
    # Define the primary and alternative file paths
    base_file_path = "../../data/results/instances/"
    alternative_path = "../data/"

    # Check if the primary path exists, use alternative if not
    if not os.path.exists(base_file_path):
        base_file_path = alternative_path


    field_location_file = os.path.join(base_file_path, f'scenario_{scenario_id}_field_location.csv')

    # Save field location as csv
    data_generator.field_locations.to_csv(field_location_file, index=True)

    # Save beet yield as a NumPy binary file (.npy)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_beet_yield.npy'), data_generator.beet_yield)

    # Save field size as a NumPy binary file (.npy)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_field_size.npy'), data_generator.field_size)

    # Save sugar concentration as a NumPy binary file (.npy)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_sugar_concentration.npy'),
            data_generator.sugar_concentration)

    # Save cost matrix as a NumPy binary file (.npy)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_distance_matrix.npy'), data_generator.cost_matrix)

    print(f"Raw Data for scenario {scenario_id} saved successfully.")


def save_machine_data(scenario_id, c_t, c_h, c_l, harvesters_data, loaders_data):

    base_file_path = "../../data/"

    # costs
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_c_t_matrix.npy'), c_t)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_c_h_matrix.npy'), c_h)
    np.save(os.path.join(base_file_path, f'scenario_{scenario_id}_c_l_matrix.npy'), c_l)

    # machine data
    with open(os.path.join(base_file_path, f'scenario_{scenario_id}_harvesters_data.pkl'), 'wb') as f:
        pickle.dump(harvesters_data, f)
    with open(os.path.join(base_file_path, f'scenario_{scenario_id}_loaders_data.pkl'), 'wb') as f:
        pickle.dump(loaders_data, f)

