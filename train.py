import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
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
from torch.utils import data
from torchvision import transforms

from pytorch_toolbelt import losses as L

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
    parser.add_argument("--epochs", "-e", default=5, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks)
        val_loss.append(loss.item())

    return np.mean(val_loss)


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

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, "train"),
        train_transforms,
        split="train",
        debug=args.debug,
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, "train"),
        train_transforms,
        split="val",
        debug=args.debug,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)

    for name, child in model.named_children():
        if name in ["fc"]:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-04,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=args.epochs
    )
    loss = L.WingLoss(width=10, curvature=2, reduction="mean")

    # 2. train & validate
    print("Ready for training...")
    for epoch in range(args.epochs):
        train_loss = train(
            model, train_dataloader, loss, optimizer, device=device, scheduler=scheduler
        )
        val_loss = validate(model, val_dataloader, loss, device=device)
        print(
            "Epoch #{:2}:\ttrain loss: {:6.3}\tval loss: {:6.3}".format(
                epoch, train_loss, val_loss
            )
        )

    # 2.1. train continued

    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 1e-6},
            {"params": model.bn1.parameters(), "lr": 1e-6},
            {"params": model.relu.parameters(), "lr": 1e-5},
            {"params": model.maxpool.parameters(), "lr": 1e-5},
            {"params": model.layer1.parameters(), "lr": 1e-4},
            {"params": model.layer2.parameters(), "lr": 1e-4},
            {"params": model.layer3.parameters(), "lr": 1e-3},
            {"params": model.layer4.parameters(), "lr": 1e-3},
            {"params": model.avgpool.parameters(), "lr": 1e-2},
            {"params": model.fc.parameters(), "lr": 1e-2},
        ],
        lr=args.learning_rate,
        weight_decay=1e-06,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    print("Ready for training again...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(
            model, train_dataloader, loss, optimizer, device=device, scheduler=scheduler
        )
        val_loss = validate(model, val_dataloader, loss, device=device)
        print(
            "Epoch #{:2}:\ttrain loss: {:6.3}\tval loss: {:6.3}".format(
                epoch, train_loss, val_loss
            )
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(f"{args.name}_best.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    if not args.debug:
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

        with open(f"submit/{args.name}_best.pth", "rb") as fp:
            best_state_dict = torch.load(fp, map_location="cpu")
            model.load_state_dict(best_state_dict)

        test_predictions = predict(model, test_dataloader, device)

        with open(f"submit/{args.name}_test_predictions.pkl", "wb") as fp:
            pickle.dump(
                {
                    "image_names": test_dataset.image_names,
                    "landmarks": test_predictions,
                },
                fp,
            )

        create_submission(args.data, test_predictions, f"submit/{args.name}_submit.csv")


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
