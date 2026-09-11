import torch
import torch.utils.data as Data
from readdata import DataReader
import numpy as np
import gc

def getDataLoader(train_file, test_file, batch_size, num_of_questions, max_step, kqn=False, sample_frac=1.0):
    handle = DataReader(train_file, test_file, max_step, num_of_questions, sample_frac=sample_frac)

    if not kqn:
        # Charger les données
        train_result = handle.getTrainData(kqn=False)
        test_result = handle.getTestData(kqn=False)

        # Cas 1 : random vectors → (data, labels)
        if isinstance(train_result, tuple):
            train_data, train_labels = train_result
            test_data, test_labels = test_result

            dtrain = torch.tensor(np.array(train_data), dtype=torch.float32)
            dtrain_labels = torch.tensor(np.array(train_labels), dtype=torch.long)

            dtest = torch.tensor(np.array(test_data), dtype=torch.float32)
            dtest_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

            train_dataset = Data.TensorDataset(dtrain, dtrain_labels)
            test_dataset = Data.TensorDataset(dtest, dtest_labels)

        # Cas 2 : one-hot → seulement data
        else:
            dtrain = torch.tensor(np.array(train_result), dtype=torch.float32)
            dtest = torch.tensor(np.array(test_result), dtype=torch.float32)

            train_dataset = Data.TensorDataset(dtrain)
            test_dataset = Data.TensorDataset(dtest)

        trainLoader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testLoader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return trainLoader, testLoader

    else:
        # Format spécial pour KQN — libération champ par champ pour réduire le pic mémoire
        train_data = handle.getTrainData(kqn=True)
        test_data  = handle.getTestData(kqn=True)

        def _to_kqn_dataset(d):
            t_in   = torch.tensor(d['in_data'],     dtype=torch.float32); del d['in_data']
            t_seq  = torch.tensor(d['seq_len'],      dtype=torch.long);    del d['seq_len']
            t_ns   = torch.tensor(d['next_skills'],  dtype=torch.long);    del d['next_skills']
            t_corr = torch.tensor(d['correctness'],  dtype=torch.float32); del d['correctness']
            t_mask = torch.tensor(d['mask'],         dtype=torch.bool);    del d['mask']
            gc.collect()
            return Data.TensorDataset(t_in, t_seq, t_ns, t_corr, t_mask)

        train_dataset = _to_kqn_dataset(train_data); del train_data; gc.collect()
        test_dataset  = _to_kqn_dataset(test_data);  del test_data;  gc.collect()

        trainLoader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testLoader  = Data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        return trainLoader, testLoader
