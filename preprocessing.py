"""Preprocessing shared by training and serving.

Training and inference must apply identical evaluation-time preprocessing. When
the two drift apart the model still runs and still returns confident answers,
it is just quietly wrong — so both `app.py` and `train.py` import from here
rather than each defining their own transform.
"""

from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Order must match the ImageFolder class order used during training.
CLASS_NAMES = ["HSIL", "LSIL", "NILM", "SCC"]


def build_eval_transform():
    """Deterministic transform used for validation, test, and serving."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform():
    """Evaluation transform plus augmentation appropriate for cytology fields.

    Cytology has no canonical orientation, so flips and rotations are safe.
    Colour jitter is kept mild: staining intensity carries diagnostic signal,
    and washing it out teaches the model to ignore something that matters.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
