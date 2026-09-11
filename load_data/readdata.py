import numpy as np
import itertools
from tqdm import tqdm

class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques, embed_dim=16, threshold=1500, sample_frac=1.0):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.sample_frac = sample_frac

    def _get_selected_records(self, file_path):
        """Return a frozenset of record indices to keep. None = keep all.
        Sampling is done at the student (block-of-3-lines) level."""
        if self.sample_frac >= 1.0:
            return None
        with open(file_path, 'r') as f:
            total_records = sum(1 for _ in f) // 3
        n_sample = max(1, round(total_records * self.sample_frac))
        rng = np.random.RandomState(42)   # fixed seed: reproducible across train/test
        return frozenset(rng.choice(total_records, n_sample, replace=False).tolist())

    def _count_total_slices(self, file_path, selected_indices=None):
        """Fast pre-pass: reads only the length lines (every 3rd) to count slices."""
        total = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 3 == 0:
                    if selected_indices is not None and (i // 3) not in selected_indices:
                        continue
                    try:
                        length = int(line.strip().strip(','))
                        total += length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                    except:
                        pass
        return total

    def get_prediction_ids(self, file_path, return_kqn_format=False):
        """Return an identifier for every prediction emitted by a model.

        A valid identifier encodes ``(original_student_index, target_step)``.
        Both standard KT models and KQN emit one prediction for each valid
        within-slice next interaction; padding has no identifier or output.
        """
        selected_indices = self._get_selected_records(file_path)
        ids_by_slice = []
        record_idx = 0

        with open(file_path, 'r') as file:
            for len_line, _, _ in itertools.zip_longest(*[file] * 3):
                if selected_indices is not None and record_idx not in selected_indices:
                    record_idx += 1
                    continue

                length = int(len_line.strip().strip(','))
                for start in range(0, length, self.maxstep):
                    steps = min(self.maxstep, length - start)
                    positions = range(steps - 1)

                    slice_ids = np.empty(len(positions), dtype=np.int64)
                    for index, position in enumerate(positions):
                        target_step = start + position + 1
                        slice_ids[index] = (np.int64(record_idx) << 32) | target_step
                    ids_by_slice.append(slice_ids)
                record_idx += 1

        if not ids_by_slice:
            return np.array([], dtype=np.int64), []
        return np.concatenate(ids_by_slice), ids_by_slice

    def getData(self, file_path, return_kqn_format=False):
        use_random_vectors = self.numofques > self.threshold

        if use_random_vectors:
            print(f"Using random vector projection (2M={2 * self.numofques}) -> embed_dim={self.embed_dim}")
            n_vectors = np.random.normal(0, 1, size=(2 * self.numofques, self.embed_dim))
        else:
            print(f"Using standard one-hot encoding (2M={2 * self.numofques})")

        # Count records (used by both paths)
        with open(file_path, 'r') as file:
            num_lines = sum(1 for _ in file)
        num_records = num_lines // 3

        # Student-level sampling: select which record indices to keep
        selected_indices = self._get_selected_records(file_path)
        if selected_indices is not None:
            print(f"Sampling {len(selected_indices)}/{num_records} students "
                  f"({100*self.sample_frac:.0f}%)")

        # ── KQN PATH: pre-allocated arrays (1× memory, no list→numpy copy) ──────
        if return_kqn_format:
            total_slices = self._count_total_slices(file_path, selected_indices)
            in_dim = self.embed_dim if use_random_vectors else 2 * self.numofques

            in_data_arr     = np.zeros((total_slices, self.maxstep, in_dim), dtype=np.float32)
            # Store skill indices (int32) instead of one-hot vectors:
            # (N, T, numofques) float32 = 46 GB for KT2 → (N, T) int32 = 3 MB
            next_skills_arr = np.zeros((total_slices, self.maxstep),          dtype=np.int32)
            correctness_arr = np.zeros((total_slices, self.maxstep),          dtype=np.float32)
            mask_arr        = np.zeros((total_slices, self.maxstep),                 dtype=bool)
            seq_len_arr     = np.zeros(total_slices,                                 dtype=np.int64)
            kqn_idx         = 0

            record_idx = 0
            with open(file_path, 'r') as file:
                for len_line, ques_line, ans_line in tqdm(
                        itertools.zip_longest(*[file] * 3),
                        total=num_records, desc="Processing records"):
                    if selected_indices is not None and record_idx not in selected_indices:
                        record_idx += 1
                        continue
                    record_idx += 1
                    length    = int(len_line.strip().strip(','))
                    questions = [int(q) for q in ques_line.strip().strip(',').split(',')]
                    answers   = [int(a) for a in ans_line.strip().strip(',').split(',')]
                    slices    = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)

                    for i in range(slices):
                        start = i * self.maxstep
                        end   = min(start + self.maxstep, length)
                        steps = end - start

                        for j in range(steps):
                            q   = questions[start + j]
                            a   = answers[start + j]
                            iid = q if a == 1 else q + self.numofques
                            if use_random_vectors:
                                in_data_arr[kqn_idx, j] = n_vectors[iid]
                            else:
                                in_data_arr[kqn_idx, j, iid] = 1.0

                            if j < steps - 1:
                                q_next = questions[start + j + 1]
                                a_next = answers[start + j + 1]
                                next_skills_arr[kqn_idx, j] = q_next  # index, not one-hot
                                correctness_arr[kqn_idx, j] = a_next
                                mask_arr[kqn_idx, j]        = True

                        seq_len_arr[kqn_idx] = steps
                        kqn_idx += 1

            print(f'done: {in_data_arr.shape}')
            return {
                'in_data':     in_data_arr,
                'next_skills': next_skills_arr,
                'correctness': correctness_arr,
                'mask':        mask_arr,
                'seq_len':     seq_len_arr,
            }

        # ── STANDARD PATH: non-KQN models ────────────────────────────────────────
        data = []
        true_labels = [] if use_random_vectors else None

        record_idx = 0
        with open(file_path, 'r') as file:
            for len_line, ques_line, ans_line in tqdm(
                    itertools.zip_longest(*[file] * 3),
                    total=num_records, desc="Processing records"):
                if selected_indices is not None and record_idx not in selected_indices:
                    record_idx += 1
                    continue
                record_idx += 1
                length    = int(len_line.strip().strip(','))
                questions = [int(q) for q in ques_line.strip().strip(',').split(',')]
                answers   = [int(a) for a in ans_line.strip().strip(',').split(',')]
                slices    = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)

                for i in range(slices):
                    start = i * self.maxstep
                    end   = min(start + self.maxstep, length)
                    steps = end - start

                    if use_random_vectors:
                        interaction = np.zeros((self.maxstep, self.embed_dim))
                    else:
                        interaction = np.zeros((self.maxstep, 2 * self.numofques))

                    for j in range(steps):
                        q   = questions[start + j]
                        a   = answers[start + j]
                        iid = q if a == 1 else q + self.numofques
                        if use_random_vectors:
                            interaction[j] = n_vectors[iid]
                        else:
                            interaction[j][iid] = 1

                    data.append(interaction.tolist())

                    if use_random_vectors:
                        true_labels.append(answers[start:end])

        print('done:', np.array(data).shape)

        if use_random_vectors:
            max_len     = max(len(x) for x in true_labels)
            true_labels = [x + [0] * (max_len - len(x)) for x in true_labels]
            return np.array(data), np.array(true_labels)
        else:
            return np.array(data)

    def getTrainData(self, kqn=False):
        print('loading train data...')
        return self.getData(self.train_path, return_kqn_format=kqn)

    def getTestData(self, kqn=False):
        print('loading test data...')
        return self.getData(self.test_path, return_kqn_format=kqn)

