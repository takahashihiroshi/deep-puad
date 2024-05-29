import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataloaders
from models import detectors, losses, trainers, utils

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="AE")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    algorithm = args.algorithm
    alpha = args.alpha if algorithm in ["PU", "PUAE", "PUSVDD"] else 0.0
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed

    # Key
    key = f"toy_{algorithm}_{alpha}_{n_epoch}_{learning_rate}_{batch_size}_{seed}"

    # Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    n_normal = 900
    n_unlabeled_anomaly = 80
    n_labeled_anomaly = 20

    train_loader = dataloaders.make_toy_data(
        batch_size=batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=True,
    )
    valid_loader = dataloaders.make_toy_data(
        batch_size=batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=True,
    )
    test_loader = dataloaders.make_toy_data(
        batch_size=batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=False,
    )

    # Model
    model: detectors.Detector
    if algorithm in ["AE", "ABC", "PUAE"]:
        model = detectors.DAE(n_in=2, n_latent=2, n_h=500, epsilon=0.1)
    elif algorithm in ["DeepSVDD", "DeepSAD", "PUSVDD"]:
        model = detectors.DeepSVDD(n_in=2, n_latent=2, n_h=500)
    else:
        # nnPU
        model = detectors.MLP(n_in=2, n_h=500)

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

    # Plot
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 20,
            "ps.useafm": True,
            "pdf.use14corefonts": True,
            "text.usetex": True,
            "font.family": "Times New Roman",
        }
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    xx = np.linspace(-7.5, 7.5, 100)
    yy = np.linspace(-7.5, 7.5, 100)
    XX, YY = np.meshgrid(xx, yy)
    PP = torch.tensor(np.dstack((XX, YY)).astype(np.float32).reshape(-1, 2))

    with torch.no_grad():
        ZZ = model.estimate(PP.to(device)).cpu().numpy().reshape(100, 100)

    plt.clf()
    fig, ax = plt.subplots()
    mappable = ax.contourf(XX, YY, ZZ, levels=20, cmap="viridis")
    fig.colorbar(mappable, format="%.2f")

    X, y = train_loader.dataset.tensors  # type: ignore
    X = X.numpy()
    y = y.numpy()

    X_unlabeled_normal = X[:n_normal]
    X_unlabeled_anomaly = X[n_normal : n_normal + n_unlabeled_anomaly]
    X_labeled_anomaly = X[-n_labeled_anomaly:]

    plt.scatter(
        X_unlabeled_normal[:, 0],
        X_unlabeled_normal[:, 1],
        color=colors[1],
        label="Unlabeled Normal",
    )
    plt.scatter(
        X_unlabeled_anomaly[:, 0],
        X_unlabeled_anomaly[:, 1],
        color=colors[3],
        label="Unlabeled Anomaly",
    )
    plt.scatter(
        X_labeled_anomaly[:, 0],
        X_labeled_anomaly[:, 1],
        color=colors[0],
        label="Labeled Anomaly",
    )

    # Plot unseen anomaly
    plt.scatter(
        [-5, 5],
        [-5, 5],
        marker="*",
        s=600,
        color=colors[4],
        edgecolors="white",
        label="Unseen Anomaly",
    )

    if algorithm == "PU":
        plt.legend(
            loc="upper left",
            handletextpad=0.15,
            borderpad=0.15,
            borderaxespad=0.15,
            framealpha=0.5,
        )

    plt.savefig(f"images/{key}_heatmap.pdf")
