import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import tqdm
from torch.utils import data
from torchvision import transforms

from utils import (
    CROP_SIZE,
    NUM_PTS,
    CropCenter,
    ScaleMinSideToSize,
    ThousandLandmarksDataset,
    TransformByKeys,
    create_submission,
    restore_landmarks_batch,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--name",
        "-n",
        help="Experiment name (for saving checkpoints and submits).",
        default="baseline",
    )
    parser.add_argument(
        "--data",
        "-d",
        help="Path to dir with target images & landmarks.",
        default="./../data/",
    )
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(
        tqdm.tqdm(loader, total=len(loader), desc="test prediction...")
    ):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape(
            (len(pred_landmarks), NUM_PTS, 2)
        )  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(
            pred_landmarks, fs, margins_x, margins_y
        )  # B x NUM_PTS x 2
        predictions[i * loader.batch_size : (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    # 1. prepare data & models
    train_transforms = transforms.Compose(
        [
            ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
            CropCenter(CROP_SIZE),
            TransformByKeys(transforms.ToPILImage(), ("image",)),
            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ("image",),
            ),
        ]
    )

    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)

    test_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, "test"),
        train_transforms,
        split="test",
        debug=args.debug,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)

    with open(f"{args.name}_test_predictions.pkl", "wb") as fp:
        pickle.dump(
            {"image_names": test_dataset.image_names, "landmarks": test_predictions}, fp
        )

    create_submission(args.data, test_predictions, f"{args.name}_submit.csv")


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
