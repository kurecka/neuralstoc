import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

import logging
logger = logging.getLogger("neuralstoc")


# Tensorflow should not allocate any GPU memory
tf.config.experimental.set_visible_devices([], "GPU")


class JaxDataset:
    def __init__(self, data, batch_size=32, shuffle=False):
        self.rng_key = jax.random.PRNGKey(42)
        self.data = jax.device_put(data)
        if shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            idx = jax.random.permutation(subkey, len(self.data))
            self.data = self.data[idx]
        
        self.data = jax.device_put(self.data)
        padding = batch_size - (len(self.data) % batch_size)
        if padding < batch_size:
            # select random elements to pad
            pad_idx = np.random.choice(len(self.data), padding)
            self.data = jnp.concatenate([self.data, self.data[pad_idx]], axis=0)

        self.data = self.data.reshape((-1, batch_size, *self.data.shape[1:]))
    
    def __iter__(self):
        return iter(self.data)

    def as_numpy_iterator(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        idx = jax.random.permutation(subkey, len(self.data))
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"JaxDataset: {len(self.data)} batches, batch_size={self.data.shape[1]}, shape={self.data.shape[2:]}"


class TrainBuffer:
    def __init__(self, max_size=2_000_000):
        """Counterexample training buffer"""
        logger.info(f"Initializing training buffer with max size {max_size}")
        self.s = []
        self.max_size = max_size
        self._cached_ds = None

    def append(self, s):
        s = np.array(s)
        self._cached_ds = None
    
        if self.max_size is None or len(self) + len(s) <= self.max_size:
            self.s.append(s)
        else:
            old_s = np.random.permutation(np.concatenate(self.s, axis=0)) if len(self.s) > 0 else np.zeros((0, *s.shape[1:]))
            new_s = np.random.permutation(s)
            all_s = np.concatenate([old_s, new_s], axis=0)
            self.s = [all_s[-self.max_size:]]

    def extend(self, lst):
        lst = [np.array(s) for s in lst]
        if len(lst) == 0:
            return
        concat = np.concatenate(lst, axis=0)
        self.append(concat)

    def clear(self):
        self.s = []
        self._cached_ds = None

    def __len__(self):
        if len(self.s) == 0:
            return 0
        return sum([s.shape[0] for s in self.s])

    @property
    def in_dim(self):
        return len(self.s[0])

    def as_tfds(self, batch_size=32):
        if self._cached_ds is not None:
            return self._cached_ds
        train_s = np.concatenate(self.s, axis=0)
        train_s = np.random.default_rng().permutation(train_s)
        train_ds = JaxDataset(train_s, batch_size=batch_size, shuffle=True)
        self._cached_ds = train_ds
        return train_ds

    def is_empty(self):
        if len(self.s) == 0:
            return True
        return False

    def get_list(self, batch_size=None):
        if len(self.s) == 0:
            return []
        if batch_size is None:
            return self.s
        else:
            new_s = []
            for s in self.s:
                for i in range(0, len(s), batch_size):
                    new_s.append(s[i : i + batch_size])
            return new_s
