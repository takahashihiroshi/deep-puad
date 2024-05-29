import argparse
import os
import random

import numpy as np
import torch

import dataloaders
from models import detectors, losses, trainers, utils

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--unseen_anomaly", type=int, default=9)
    parser.add_argument("--algorithm", type=str, default="AE")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    normal_class = args.normal_class
    unseen_anomaly = args.unseen_anomaly
    algorithm = args.algorithm
    alpha = args.alpha if algorithm in ["PU", "PUAE", "PUSVDD"] else 0.0
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed

    # Key
    key = f"{dataset}_{normal_class}_{unseen_anomaly}_{algorithm}_{alpha}_{n_epoch}_{learning_rate}_{batch_size}_{seed}"

    # Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_loader, valid_loader, test_loader = dataloaders.load(
        name=dataset,
        batch_size=batch_size,
        normal_class=normal_class,
        unseen_anomaly=unseen_anomaly,
    )

    # Deep Model
    n_in = 32 * 32
    n_latent = 40
    n_h = 1000

    model: detectors.Detector
    if algorithm in ["AE", "ABC", "PUAE"]:
        model = detectors.Autoencoder(n_in=n_in, n_latent=n_latent, n_h=n_h)
    elif algorithm in ["DeepSVDD", "DeepSAD", "PUSVDD"]:
        model = detectors.DeepSVDD(n_in=n_in, n_latent=n_latent, n_h=n_h)
    else:
        # nnPU
        model = detectors.MLP(n_in=n_in, n_h=n_h)

    model = model.to(device)
    print(type(model))

    criterion: losses.Loss
    match algorithm:
        case "AE" | "DeepSVDD":
            criterion = losses.AELoss()
        case "ABC":
            criterion = losses.ABCLoss()
        case "DeepSAD":
            criterion = losses.DeepSADLoss()
        case "PUAE":
            criterion = losses.PUAELoss(alpha=alpha)
        case "PUSVDD":
            criterion = losses.PUSVDDLoss(alpha=alpha)
        case _:  # nnPU
            criterion = losses.PULoss(alpha=alpha)

    # Train
    for path in ["checkpoints", "results", "images"]:
        os.makedirs(path, exist_ok=True)

    if isinstance(model, detectors.DeepSVDD):
        # Pre-Training as Autoencoder
        print("Pre-Training:")
        pre_trainer = trainers.Trainer(device)
        pre_trainer.fit(
            model=model,
            criterion=losses.AELoss(),
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=f"checkpoints/{key}.pt",
            n_epoch=n_epoch,
            learning_rate=learning_rate,
            weight_decay=1e-3,
        )
        utils.set_center(model=model, train_loader=train_loader, device=device, eps=0.1)

    print("Training:")
    trainer = trainers.Trainer(device=device)
    trainer.fit(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        checkpoint=f"checkpoints/{key}.pt",
        n_epoch=n_epoch,
        learning_rate=learning_rate,
        weight_decay=1e-3,
    )

    # Test
    test_score = utils.compute_auc(model=model, test_loader=test_loader, device=device)
    print("AUC:", test_score)

    result = utils.Result(
        train_loss=trainer.train_losses,
        valid_loss=trainer.valid_losses,
        test_score=test_score,
    )

    # Save and Plot Results
    utils.save_result(result=result, file_name=f"results/{key}.json")
    utils.plot_result(result=result, file_name=f"images/{key}.pdf")
