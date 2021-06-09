from tqdm import tqdm

TQDMM_KWARGS = {"ncols": 99, "leave": False}


def tqdmm(iterable, **kwargs):
    return tqdm(iterable, **TQDMM_KWARGS, **kwargs)
