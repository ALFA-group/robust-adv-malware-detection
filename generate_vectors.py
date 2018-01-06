import os
import pickle

from datasets.datasets import load_data
from utils.utils import load_parameters

malicious_vector_filepath = "/home/alexmalmeng/saved_feature_vectors/malicious/"
benign_vector_filepath = "/home/alexmalmeng/saved_feature_vectors/benign/"

parameters = load_parameters("parameters.ini")

assertion_message = "Flag must be on to generate. Changes return of PortableExecutableDataset."
assert eval(parameters['dataset']['generate_feature_vector_files']) is True, assertion_message

train_dataloader_dict, valid_dataloader_dict, test_dataloader_dict, num_features = load_data(
    parameters)

print(
    len(train_dataloader_dict['malicious'].dataset) + len(test_dataloader_dict['malicious'].dataset)
)
print(len(train_dataloader_dict['benign'].dataset) + len(test_dataloader_dict['benign'].dataset))

for data_dict in [train_dataloader_dict, valid_dataloader_dict, test_dataloader_dict]:

    for filetype in data_dict:
        print(filetype)
        dataloader = data_dict[filetype]

        for index, data in enumerate(dataloader):
            print(index, filetype)
            vector, label, filepath = data

            filename = filepath[0].split("/")[-1]

            if filetype == 'malicious':
                pickle.dump(vector, open(os.path.join(malicious_vector_filepath, filename), 'wb'))
            else:
                pickle.dump(vector, open(os.path.join(benign_vector_filepath, filename), 'wb'))
