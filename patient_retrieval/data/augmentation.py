import numpy as np
import torch


class TimeSeriesAugmentor:
    """

    Args:
        scale_range     : (min, max) scaling factor (default: (0.8, 1.2))
    """

    def __init__(
        self,
        crop_ratio: float  = 0.5,
        mask_ratio: float  = 0.15,
        scale_range: tuple = (0.8, 1.2),
        jitter_std: float  = 0.05,
    ):
        self.crop_ratio  = crop_ratio
        self.mask_ratio  = mask_ratio
        self.scale_range = scale_range
        self.jitter_std  = jitter_std

    def random_crop(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, D)
        """
        T, D = x.shape
        min_len = max(1, int(T * self.crop_ratio))
        crop_len = np.random.randint(min_len, T + 1)
        start = np.random.randint(0, T - crop_len + 1)

        out = np.zeros_like(x)
        out[:crop_len] = x[start:start + crop_len]
        return out

    def random_masking(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, D)
        """
        T, D = x.shape
        mask = np.random.rand(T) < self.mask_ratio
        out = x.copy()
        out[mask, :] = 0.0
        return out

    def random_scaling(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, D)
        """
        _, D = x.shape
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(1, D))
        return x * scale

    def gaussian_jitter(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*x.shape) * self.jitter_std
        obs_mask = (x != 0).astype(float)
        return x + noise * obs_mask

    def __call__(self, x: np.ndarray) -> tuple:
        """
        x: (T, D)
        """
        view1 = self.random_crop(x)
        view1 = self.random_masking(view1)
        view1 = self.random_scaling(view1)
        view1 = self.gaussian_jitter(view1)

        view2 = self.random_crop(x)
        view2 = self.random_masking(view2)
        view2 = self.random_scaling(view2)
        view2 = self.gaussian_jitter(view2)

        return view1.astype(np.float32), view2.astype(np.float32)


class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    """

      view1, view2: (T, D) tensor — augmented ts_rich
    """

    def __init__(self, base_dataset, augmentor: TimeSeriesAugmentor = None):
        self.ds  = base_dataset
        self.aug = augmentor or TimeSeriesAugmentor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        ts_np  = sample["ts_rich"].numpy()

        v1, v2 = self.aug(ts_np)

        return {
            "view1"      : torch.tensor(v1),
            "view2"      : torch.tensor(v2),
            "label"      : sample["label"],
            "sofa_proxy" : sample["sofa_proxy"],
            "stay_id"    : sample["stay_id"],
        }