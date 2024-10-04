from unittest.mock import call
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse
from torch.utils.data import DataLoader

from src.data import chaksu, FIVES, cityscapes
from src.models import PHISeg, ProbUNet, UNet, UNetMCDropout
from src.augmentation import apply_augmentation

#Chaksu = "/mnt/qb/work/baumgartner/bkc562/ResearchProject/Chaksu/Chaksu_ROI_binary.h5"
#RIGA = "/mnt/qb/work/baumgartner/bkc562/ResearchProject/RIGA-dataset/RIGA_ROI.h5"
FIVES_path = "/mnt/qb/work/baumgartner/bkc562/MasterThesis/FIVES.h5"
#cityscapes_path = '/mnt/qb/work/baumgartner/bkc562/MasterThesis/cityscapes_group.h5'


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store",
        default=0,
        type=int,
        help="Random seed for pl.seed_everything function.",
    )
    parser.add_argument(
        "--method",
        dest="method",
        action="store",
        default=None,
        type=str,
        help="The method should be used [probunet, phiseg, unet, unet-mcdropout]",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        action="store",
        default=16,
        type=int,
        help="Batch size for training.",
    )
    args = parser.parse_args()

    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [
            git_hash,
            f"seed={args.random_seed}",
            args.method,
            human_readable_extra,
            f"bs={args.batch_size}",
        ]
    )

    pl.seed_everything(seed=args.random_seed)

    train_dataset = FIVES.Fives(file_path=FIVES_path, t="train", transform=apply_augmentation)
    valid_dataset = FIVES.Fives(file_path=FIVES_path, t="val", transform=None)

    #train_dataset = cityscapes.Cityscapes(file_path=cityscapes_path, t="train", transform=None)
    #valid_dataset = cityscapes.Cityscapes(file_path=cityscapes_path, t="val", transform=None)

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    logger = TensorBoardLogger(
        save_dir="./runsFIVES", name=experiment_name, default_hp_metric=False
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            filename="best-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val/dice",
            filename="best-dice-{epoch}-{step}",
            mode="max",
        ),
        ModelCheckpoint(
            monitor="val/squared error pp",
            filename="best-pp-{epoch}-{step}",
            mode="min",
        ),
    ]

    if args.method == "phiseg":
        model = PHISeg(
            total_levels=7, latent_levels=5, zdim=2, num_classes=2, beta=1.0, mixup=False
        )
    elif args.method == "probunet":
        model = ProbUNet(
            total_levels=7, zdim=6, num_classes=2, beta=1.0, mixup=False
        )
    elif args.method == "unet-mcdropout":
        model = UNetMCDropout(
            total_levels=7, num_classes=2, mixup=False
        )
    elif args.method == "unet":
        model = UNet(
            total_levels=7, num_classes=2, mixup=False
        )
    else:
        raise ValueError(f"Unknown method: {args.method}.")

    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.25,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
