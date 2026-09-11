"""
Usage:
    n_runs.py --hidden=<h> [options]

Options:
    --lr=<float>        learning rate [default: 0.001]
    --bs=<int>          batch size [default: 64]
    --seed=<int>        random seed [default: 59]
    --epochs=<int>      number of epochs [default: 10]
    --cuda=<int>        use GPU id [default: 0]
    --hidden=<int>      hidden dimension [default: 128]
    --layers=<int>      layers [default: 1]
    --heads=<int>       transformer heads [default: 8]
    --dropout=<float>   dropout [default: 0.1]
"""

import os
import sys
import gc
import random
import torch
import json
import numpy as np
import time
import psutil
from docopt import docopt
from codecarbon import EmissionsTracker
from thop import profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "load_data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from dataloader import getDataLoader
from readdata import DataReader
from evaluation import train_epoch, test_epoch, lossFunc, get_predictions



# Utils


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_system_metrics():
    return psutil.cpu_percent(), psutil.virtual_memory().percent


def compute_flops(model, questions, length, device, model_type='RNN'):
    input_dim = 16 if questions > 1500 else 2 * questions
    try:
        if model_type == 'KQN':
            dummy_in = torch.zeros(1, length, input_dim, dtype=torch.float32).to(device)
            dummy_seq = torch.tensor([length], dtype=torch.long).to(device)
            dummy_skills = torch.zeros(1, length, questions, dtype=torch.float32).to(device)
            flops, _ = profile(model, inputs=(dummy_in, dummy_seq, dummy_skills), verbose=False)
        else:
            dummy_input = torch.zeros(1, length, input_dim, dtype=torch.float32).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops
    except:
        return None



# Model factory


def create_model(model_type, questions, hidden, layers, heads, dropout, length, device):

    if questions > 1500:
        input_dim = 16
        compressed = True
    else:
        input_dim = 2 * questions
        compressed = False

    if model_type == 'RNN':
        from RNNModel import RNNModel
        return RNNModel(input_dim, hidden, layers, questions, device)

    elif model_type == 'LSTM':
        from RNNModel import LSTMModel
        return LSTMModel(input_dim, hidden, layers, questions, device)

    elif model_type == 'SAKT':
        from SAKTModel import SAKTModel
        return SAKTModel(heads, length, hidden, questions, dropout, input_dim=input_dim)
    
    elif model_type == 'FCN':
        from FCNModel import FCNModel
        return FCNModel(input_dim, hidden, layers, questions, device), compressed

    elif model_type == 'DKVMN':
        from DKVMNModel import DKVMNModel
        # memory_size = latent concepts (capped at 50, as in the DKVMN paper)
        # For compressed datasets only pred[:,:,0] is used → output_dim=1 saves GPU memory
        dkvmn_memory = min(questions, 50)
        dkvmn_output = 1 if compressed else questions
        return DKVMNModel(input_dim, dkvmn_memory, 50, dkvmn_output, device)

    elif model_type == 'DKTPlus':
        from DKTPlusModel import DKTPlusModel
        return DKTPlusModel(input_dim, hidden, layers, questions, device, lambda_w1=0.5, lambda_w2=0.5, lambda_o=0.5), compressed

    elif model_type == 'KQN':
        from KQNModel import KQN
        return KQN(n_skills=questions, n_hidden=hidden, n_rnn_hidden=hidden, n_mlp_hidden=hidden, n_rnn_layers=layers, rnn_type='lstm', device=device, rnn_input_dim=input_dim), compressed

    else:
        raise ValueError(model_type)



# MAIN


def main():
    args = docopt(__doc__)

    os.makedirs("emissions", exist_ok=True)

    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    base_dir = "dataset"
    model_types = ['RNN','LSTM','FCN','SAKT','DKVMN','DKTPlus','KQN'] 
    n_runs = 3

    for model_type in model_types:
        print(f"\n### Model: {model_type} ###")

        all_results = []

        for dataset in ['KT2', 'KT3', 'KT4','nips_task_3_4']: #os.listdir(base_dir):
            print(f"\nDataset: {dataset}")

            dataset_path = os.path.join(base_dir, dataset)
            train_path = os.path.join(dataset_path, "builder_train.csv")
            test_path = os.path.join(dataset_path, "builder_test.csv")

            q_ids = load_question_ids(train_path)
            questions = max(max(seq) for seq in q_ids) + 1

            if dataset in ('statics','assistChall'):
                length=500
            elif dataset=='assist2009':
                length=500
            elif dataset=='synthetic':
                length=50
            else:
                length=100

            # Sous-échantillonnage par bloc d'étudiants pour les grands datasets
            if dataset in ('KT2', 'KT3', 'KT4'):
                _sample_frac = 0.1
            elif dataset == 'nips_task_3_4':
                _sample_frac = 0.2
            else:
                _sample_frac = 1.0

            trainLoader, testLoader = getDataLoader(train_path, test_path, bs, questions, length,
                                                     kqn=(model_type=='KQN'),
                                                     sample_frac=_sample_frac)

            # Val loader: last 20% of train sequences (for stacking meta-learner)
            _train_ds = trainLoader.dataset
            _n_total = len(_train_ds)
            _n_val = max(1, int(_n_total * 0.2))
            _val_indices = list(range(_n_total - _n_val, _n_total))
            _val_subset = torch.utils.data.Subset(_train_ds, _val_indices)
            valLoader = torch.utils.data.DataLoader(_val_subset, batch_size=bs, shuffle=False)

            # Store interaction identifiers with predictions so ensemble methods
            # can align models by the same target interaction, not array offset.
            _id_reader = DataReader(train_path, test_path, length, questions,
                                    sample_frac=_sample_frac)
            _test_ids, _ = _id_reader.get_prediction_ids(
                test_path, return_kqn_format=(model_type == 'KQN')
            )
            _, _train_ids_by_slice = _id_reader.get_prediction_ids(
                train_path, return_kqn_format=(model_type == 'KQN')
            )
            _val_ids = np.concatenate(_train_ids_by_slice[-_n_val:])

            # ===== STORE RUNS =====
            runs_metrics = []

            for run in range(n_runs):
                print(f"Run {run+1}/{n_runs}")

                setup_seed(seed + run)

                result = create_model(model_type, questions, hidden, layers, heads, dropout, length, device)
                model = result[0] if isinstance(result, tuple) else result
                model_size = sum(p.numel() for p in model.parameters())

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_func = lossFunc(questions, length, device, compressed=(questions > 1500))

                # ===== TRAINING =====
            

                training_time = 0
                for epoch in range(epochs):
                    start = time.time()
                    model, optimizer = train_epoch(model, trainLoader, optimizer, loss_func, device, model_type)
                    training_time += time.time() - start

          
            

                # ===== INFERENCE =====
    

                model.eval()
                start = time.time()
                auc, acc, f1 = test_epoch(model, testLoader, loss_func, device, model_type)
                inference_time = time.time() - start


                # ===== SAVE MODEL WEIGHTS =====
                os.makedirs(f"saved_models/{dataset}", exist_ok=True)
                torch.save(model.state_dict(), f"saved_models/{dataset}/{model_type}_run{run}.pt")

                # ===== SAVE RAW PREDICTIONS (test + val) =====
                os.makedirs(f"saved_preds/{dataset}", exist_ok=True)
                _test_preds, _test_labels = get_predictions(model, testLoader, loss_func, device, model_type)
                if len(_test_preds) != len(_test_ids):
                    raise RuntimeError(
                        f"Prediction/ID length mismatch for {model_type} on {dataset} test: "
                        f"{len(_test_preds)} predictions vs {len(_test_ids)} IDs"
                    )
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_test_preds.npy", _test_preds)
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_test_ids.npy", _test_ids)
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_test_labels.npy", _test_labels)
                np.save(f"saved_preds/{dataset}/ground_truth_run{run}_test.npy", _test_labels)

                _val_preds, _val_labels = get_predictions(model, valLoader, loss_func, device, model_type)
                if len(_val_preds) != len(_val_ids):
                    raise RuntimeError(
                        f"Prediction/ID length mismatch for {model_type} on {dataset} validation: "
                        f"{len(_val_preds)} predictions vs {len(_val_ids)} IDs"
                    )
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_val_preds.npy", _val_preds)
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_val_ids.npy", _val_ids)
                np.save(f"saved_preds/{dataset}/{model_type}_run{run}_val_labels.npy", _val_labels)
                np.save(f"saved_preds/{dataset}/ground_truth_run{run}_val.npy", _val_labels)

    
                runs_metrics.append({
                    "auc": auc,
                    "f1": f1,
                    "accuracy": acc,
                    "training_time": training_time,
                    "inference_time": inference_time,
                })

                # ===== LIBÉRER MÉMOIRE (fin de run) =====
                del model, optimizer, loss_func
                gc.collect()
                torch.cuda.empty_cache()

            # ===== AVERAGE =====
            avg = {}
            for key in runs_metrics[0]:
                values = [r[key] for r in runs_metrics if r[key] is not None]
                avg[f"avg_{key}"] = np.mean(values)
                avg[f"std_{key}"] = np.std(values)

            avg["dataset"] = dataset
            avg["model"] = model_type

            all_results.append(avg)

            # ===== SAVE INDIVIDUAL RESULTS =====
            os.makedirs(f"results/individual/{dataset}", exist_ok=True)
            ind_result = {
                "model": model_type,
                "dataset": dataset,
                "n_runs": n_runs,
                "avg_auc":            avg.get("avg_auc", 0),
                "std_auc":            avg.get("std_auc", 0),
                "avg_f1":             avg.get("avg_f1", 0),
                "std_f1":             avg.get("std_f1", 0),
                "avg_accuracy":       avg.get("avg_accuracy", 0),
                "std_accuracy":       avg.get("std_accuracy", 0),
                "avg_training_time":  avg.get("avg_training_time", 0),
                "avg_inference_time": avg.get("avg_inference_time", 0),
            }
            with open(f"results/individual/{dataset}/{model_type}.json", "w") as f:
                json.dump(ind_result, f, indent=4)

            # ===== LIBÉRER MÉMOIRE (fin de dataset) =====
            del trainLoader, testLoader, valLoader, _val_subset
            gc.collect()
            torch.cuda.empty_cache()

        os.makedirs("final_results", exist_ok=True)

        with open(f"final_results/{model_type}.json", "w") as f:
            json.dump(all_results, f, indent=4)

    print("\n DONE")



# REQUIRED FUNCTION

# def load_question_ids(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     q_ids = []
#     for i in range(0, len(lines), 3):
#         try:
#             n = int(lines[i].strip())
#             ids = list(map(int, filter(None, lines[i+1].strip().split(','))))
#             q_ids.append(ids[:n])
#         except:
#             continue
#     return q_ids

def load_question_ids(file_path):
    with open(file_path, 'r') as file:
         lines = file.readlines() 
    q_ids = []
    for i in range(0, len(lines), 3):
        try:
            n_steps = int(lines[i].strip())
            question_ids = [int(q) for q in lines[i + 1].strip().split(',') if q.strip()] 
            q_ids.append(question_ids[:n_steps])
        except:
            continue
    return q_ids


if __name__ == '__main__':
    main()