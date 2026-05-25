def apply_augmentation(example):
    # Reserved for photometric/geometric training augmentation. Kept as a no-op
    # in this transplant so the dataset contract stays stable from day one.
    return example
