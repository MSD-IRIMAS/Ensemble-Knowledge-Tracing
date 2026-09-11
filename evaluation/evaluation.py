import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np

def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(),
                          torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(),
                                  torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().cpu().numpy(),
        torch.round(prediction).detach().cpu().numpy())
    
    print(f"auc: {auc:.2f} | f1: {f1:.4f} | precision: {precision:.2f}")

    return round(auc,2), round(precision,2), round(f1,2)


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device, compressed=True):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device
        self.compressed = compressed

    def forward(self, pred, batch,true_labels=None):
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(pred.shape[0]):
            if not self.compressed:
                delta = batch[student][:, 0:self.num_of_questions] + batch[
                    student][:, self.num_of_questions:]  # shape: [length, questions]
                
                temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
                index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                    dtype=torch.long, device=self.device)
                p = temp.gather(0, index)[0]
                a = (((batch[student][:, 0:self.num_of_questions] -
                    batch[student][:, self.num_of_questions:]).sum(1) + 1) //
                    2)[1:]
                valid_targets = delta[1:].sum(dim=1) > 0
                p = p[valid_targets]
                a = a[valid_targets]
            else:
                temp = pred[student][:self.max_step - 1]
                p = temp[:, 0]
                # As with the one-hot path, prediction at t is evaluated
                # against the outcome observed at t + 1.
                a = true_labels[student][1:self.max_step]

                # Padded steps have an all-zero input row; use that (not the
                # model output) to find the true valid length of the sequence.
                valid_steps = int((batch[student].abs().sum(dim=1) > 0).sum().item())
                valid_len = max(valid_steps - 1, 0)
                p = p[:valid_len]
                a = a[:valid_len]

            p = p.float()
            a = a.float()
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func, device, model_type='RNN'):
    model.to(device).float()
    model.train()

    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        optimizer.zero_grad()

        if model_type == 'KQN':
            in_data, seq_len, next_skills, correctness, mask = [x.to(device) for x in batch]
            pred = model(in_data, seq_len, next_skills)
            loss = model.loss(pred, correctness, mask)

        elif model_type == 'LITE':
            batch = batch.to(device)
            pred = model(batch)
            loss, _, _ = loss_func(pred, batch)

        else:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, labels = [x.to(device) for x in batch]
                pred = model(data)

                if hasattr(loss_func, "compressed") and loss_func.compressed:
                    loss, _, _ = loss_func(pred, data, true_labels=labels)
                else:
                    loss, _, _ = loss_func(pred, labels)
            else:
                if isinstance(batch, (list, tuple)):
                    data = batch[0].to(device)
                else:
                    data = batch.to(device)

                pred = model(data)
                loss, _, _ = loss_func(pred, data)

        loss.backward()
        optimizer.step()

    return model, optimizer



def test_epoch(model, testLoader, loss_func, device, model_type='RNN'):
    model.to(device).float()
    model.eval()

    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)

    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        if model_type == 'KQN':
            in_data, seq_len, next_skills, correctness, mask = [x.to(device) for x in batch]
            pred = model(in_data, seq_len, next_skills)
            pred = torch.sigmoid(pred)

            if pred.dim() == 3:
                pred = pred.squeeze(-1)

            if mask.shape != pred.shape:
                pad_len = pred.shape[1] - mask.shape[1]
                if pad_len > 0:
                    mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                elif pad_len < 0:
                    mask = mask[:, :pred.shape[1]]

            if correctness.shape != pred.shape:
                pad_len = pred.shape[1] - correctness.shape[1]
                if pad_len > 0:
                    correctness = torch.nn.functional.pad(correctness, (0, pad_len), value=0)
                elif pad_len < 0:
                    correctness = correctness[:, :pred.shape[1]]

            preds = pred.masked_select(mask)
            labels = correctness.masked_select(mask)

        elif model_type == 'LITE':
            batch = batch.to(device)
            pred = model(batch)
            loss, preds, labels = loss_func(pred, batch)

        else:
            with torch.no_grad():
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, labels = [x.to(device) for x in batch]
                    pred = model(data)

                    if hasattr(loss_func, "compressed") and loss_func.compressed:
                        _, preds, labels = loss_func(pred, data, true_labels=labels)
                    else:
                        _, preds, labels = loss_func(pred, labels)
                        
                else:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0].to(device)
                    else:
                        data = batch.to(device)

                    pred = model(data)
                    _, preds, labels = loss_func(pred, data)


                torch.cuda.empty_cache()

        prediction = torch.cat([prediction, preds])
        ground_truth = torch.cat([ground_truth, labels])

    auc, acc, f1 = performance(ground_truth, prediction)
    return auc, acc, f1


def get_predictions(model, loader, loss_func, device, model_type='RNN'):
    """
    Run inference and return (predictions, ground_truth) as numpy arrays.
    Used for saving raw outputs needed by ensemble methods.
    """
    model.to(device).float()
    model.eval()

    all_preds = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device)

    with torch.no_grad():
        for batch in loader:
            if model_type == 'KQN':
                in_data, seq_len, next_skills, correctness, mask = [x.to(device) for x in batch]
                pred = model(in_data, seq_len, next_skills)
                pred = torch.sigmoid(pred)
                if pred.dim() == 3:
                    pred = pred.squeeze(-1)
                if mask.shape != pred.shape:
                    pad_len = pred.shape[1] - mask.shape[1]
                    if pad_len > 0:
                        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                    elif pad_len < 0:
                        mask = mask[:, :pred.shape[1]]
                if correctness.shape != pred.shape:
                    pad_len = pred.shape[1] - correctness.shape[1]
                    if pad_len > 0:
                        correctness = torch.nn.functional.pad(correctness, (0, pad_len), value=0)
                    elif pad_len < 0:
                        correctness = correctness[:, :pred.shape[1]]
                preds = pred.masked_select(mask)
                labels = correctness.masked_select(mask)
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, true_labels = [x.to(device) for x in batch]
                    pred = model(data)
                    if hasattr(loss_func, 'compressed') and loss_func.compressed:
                        _, preds, labels = loss_func(pred, data, true_labels=true_labels)
                    else:
                        _, preds, labels = loss_func(pred, true_labels)
                else:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0].to(device)
                    else:
                        data = batch.to(device)
                    pred = model(data)
                    _, preds, labels = loss_func(pred, data)

            all_preds = torch.cat([all_preds, preds])
            all_labels = torch.cat([all_labels, labels])

    return all_preds.cpu().numpy(), all_labels.cpu().numpy()

