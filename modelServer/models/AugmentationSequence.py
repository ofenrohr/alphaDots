import numpy as np
from keras.utils import Sequence


class AugmentationSequence(Sequence):

    def __init__(self, input_data, policy_data, value_data, batch_size, input_postprocessing=lambda x:x,
                 policy_postprocessing=lambda x:x, value_postprocessing=lambda x:x):
        self.input, self.policy, self.value = input_data, policy_data, value_data
        self.batch_size = batch_size
        self.input_postprocessing = input_postprocessing
        self.policy_postprocessing = policy_postprocessing
        self.value_postprocessing = value_postprocessing

    def __len__(self):
        return int(np.ceil(self.input.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_input = self.input[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_policy = self.policy[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_value = self.value[idx * self.batch_size : (idx+1) * self.batch_size]

        augmented_batch_input, augmented_batch_policy = self.augment_data(batch_input, batch_policy)

        return self.input_postprocessing(augmented_batch_input), [ self.policy_postprocessing(augmented_batch_policy),
               self.value_postprocessing(batch_value)]

    def augment_data(self, input_data, policy_data):
        assert (len(input_data.shape) == 3)
        assert (len(policy_data.shape) == 3)
        assert (input_data.shape[0] == policy_data.shape[0])
        input_ret = np.zeros(input_data.shape, dtype=input_data.dtype)
        policy_ret = np.zeros(policy_data.shape, dtype=policy_data.dtype)
        for i in range(input_data.shape[0]):
            op = np.random.randint(5)
            if op == 0:
                input_ret[i] = input_data[i]
                policy_ret[i] = policy_data[i]
            if op == 1:
                input_ret[i] = np.flip(input_data[i], -1)
                policy_ret[i] = np.flip(policy_data[i], -1)
            if op == 2:
                input_ret[i] = np.flip(input_data[i], -2)
                policy_ret[i] = np.flip(policy_data[i], -2)
            if op == 3:
                input_ret[i] = np.flip(np.flip(input_data[i], -1), -2)
                policy_ret[i] = np.flip(np.flip(policy_data[i], -1), -2)
            if op == 4:
                input_ret[i] = np.flip(np.flip(input_data[i], -2), -1)
                policy_ret[i] = np.flip(np.flip(policy_data[i], -2), -1)
        return input_ret, policy_ret


    def shuffle_in_unison(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def on_epoch_end(self):
        self.shuffle_in_unison(self.input, self.policy, self.value)

