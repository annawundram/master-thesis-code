import gc
from src.models import PHISeg, ProbUNet, UNet, UNetMCDropout
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from torch.utils.data import DataLoader
from src.data import cityscapes


def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient for a given pair of ground truth and predicted segmentation masks. Mean over labels.

    Parameters:
    - y_true (numpy.ndarray): Ground truth segmentation mask.
    - y_pred (numpy.ndarray): Predicted segmentation mask.

    Returns:
    - float: The Dice coefficient.
    """
    batch_size = y_true.shape[0]
    n = y_pred.shape[1]
    batch_dice_scores = []
    c_classes = 8

    for i in range(batch_size):
        y_t = y_true[i]
        dice_scores = []
        for j in range(n):
            y_p = y_pred[i, j]
            dice = 0
            for value in list(range(c_classes)):
                true_binary = y_t == value
                pred_binary = y_p == value
                intersection = np.logical_and(true_binary, pred_binary)
                if true_binary.sum() + pred_binary.sum() == 0:
                    dice += 1
                    continue
                class_dice = (2 * intersection.sum()) / (true_binary.sum() + pred_binary.sum())
                dice += class_dice
            dice = dice / c_classes
            dice_scores.append(dice)
        batch_dice_scores.append(np.mean(dice_scores))
    return batch_dice_scores


def predict_dice(mat):
    """
        Compute the predicted Dice coefficient for a given predicted segmentation mask by calculating a confusion
        matrix.

        Parameters:
        - mat matrix of predicted segmentation with batch and samples

        Returns:
        - float: The Dice coefficient.
        """
    batch_size = mat.shape[0]
    n = mat.shape[1]
    c_classes = 8
    mean_dice_batch = []
    var_dice_batch = []

    for i in range(batch_size):
        dice_scores = []
        for j in range(n):
            m = mat[i,j]
            dice = 0
            for c in range(c_classes):
                positives = m[c] > 0.5
                negatives = m[c] <= 0.5

                TP = m[c][positives].sum()
                if TP == 0:
                    dice_class = 1
                    dice += dice_class
                    continue
                FP = positives.sum() - TP
                FN = m[c][negatives].sum()
                # TN = negatives.sum() - FN
                dice_class = (2 * TP) / (2 * TP + FP + FN)
                dice += dice_class
            dice = dice/c_classes
            dice_scores.append(dice)
        mean_dice_batch.append(np.mean(dice_scores))
        var_dice_batch.append(np.var(dice_scores))

    return mean_dice_batch, var_dice_batch


def get_samples(model, input, n_samples, temp):
    if model.__class__.__name__ == "UNet":
       model_pred = model.predict(input)

       # temperature scale
       model.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1) * temp)
       model_pred = model.temperature_scaling(model_pred)

       s = torch.nn.Softmax(dim=1)
       samples_s = s(model_pred)  # 4xcx320x320
       samples_s = samples_s.detach().cpu().numpy()

       samples = torch.argmax(model_pred, dim=1)  # 4x320x320

       samples = samples.detach().cpu().numpy()
       samples = samples.astype("uint8")

       return samples, samples_s

    else:
        model_pred = model.predict_output_samples(input, N=n_samples)  # 4xnxcx320x320

        # temperature scale
        model.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1) * temp)
        model_pred = model.temperature_scaling(model_pred)

        s = torch.nn.Softmax(dim=2)
        samples_s = s(model_pred)  # 4xnxcx320x320
        samples_s = samples_s.detach().cpu().numpy()  # 4xnxcx320x320
        samples = torch.argmax(model_pred, dim=2)  # 4xnx320x320

        samples = samples.detach().cpu().numpy()
        samples = samples.astype("uint8")

        return samples, samples_s


def predict(image, n_samples, model, temp):
    """
        Predict n samples for a given sample

        Parameters:
        - image Image for which to predict
        - n_samples Number of samples to draw from model
        - model Model that predicts image
        - temp Actual temp. scaling

        Returns:
        - samples hardened samples in shape 4 x n_samples x 320 x 320
        - samples_s softmax samples in shape 4 x n_samples x c x 320 x 320
        """
    return get_samples(model, image, n_samples, temp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating trained models."
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="Path to directory containing the checkpoint files. The most recent checkpoint will be loaded.",
        nargs='+',
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples to be drawn",
        default=100
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Either 'cpu' or 'gpu'",
        default='cpu',
        required=True
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoints_dir

    device = args.device

    # data
    cityscapes_path = '/mnt/qb/work/baumgartner/bkc562/MasterThesis/cityscapes_group.h5'
    cal_dataset = cityscapes.Cityscapes(file_path=cityscapes_path, t="train", transform=None)
    test_dataset = cityscapes.Cityscapes(file_path=cityscapes_path, t="val", transform=None)
    cal_loader = DataLoader(cal_dataset, batch_size=4, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, drop_last=False, shuffle=False)

    # Model
    if "phiseg" in checkpoint_dir[0]:
        model = PHISeg.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        temp = 1
        model.eval()
        saving_name = "phisegcityscapesmulticlass"
    elif "probunet" in checkpoint_dir[0]:
        model = ProbUNet.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        temp = 1
        model.eval()
        saving_name = "probunetcityscapesmulticlass"
    elif "mcdropout" in checkpoint_dir[0]:
        model = UNetMCDropout.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        temp = 1
        model.eval()
        saving_name = "mcdropoutcityscapesmulticlass"
    else:
        model = None
        model1 = UNet.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        model2 = UNet.load_from_checkpoint(checkpoint_dir[1], map_location=torch.device(device))
        model3 = UNet.load_from_checkpoint(checkpoint_dir[2], map_location=torch.device(device))
        model4 = UNet.load_from_checkpoint(checkpoint_dir[3], map_location=torch.device(device))
        model5 = UNet.load_from_checkpoint(checkpoint_dir[4], map_location=torch.device(device))
        model6 = UNet.load_from_checkpoint(checkpoint_dir[5], map_location=torch.device(device))
        model7 = UNet.load_from_checkpoint(checkpoint_dir[6], map_location=torch.device(device))
        model8 = UNet.load_from_checkpoint(checkpoint_dir[7], map_location=torch.device(device))
        model9 = UNet.load_from_checkpoint(checkpoint_dir[8], map_location=torch.device(device))
        model10 = UNet.load_from_checkpoint(checkpoint_dir[9], map_location=torch.device(device))

        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        model8.eval()
        model9.eval()
        model10.eval()

        saving_name = "unetcityscapesmulticlass"

    epochs = 1  # number of epochs
    n = len(cal_dataset)
    n_samples = args.n_samples  # number of samples from model
    alpha = 0.1  # conformal prediction
    temps = [2.005, 1.886, 1.757, 1.966, 1.924, 1.775, 2.011, 1.994, 1.959, 1.964]  # for cityscapes ensemble

    # ------------- calculate on calibration set -------------
    gt_dice = []
    pred_dice = []
    pred_var = []
    for epoch in range(epochs):
        for images, labels in cal_loader:
            labels = labels.detach().cpu().numpy()
            if len(checkpoint_dir) > 1:  # ensemble
                ensemble = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
                samples = []
                samples_s = []
                for temp, m in zip(temps, ensemble):
                    sample, sample_s = predict(images, n_samples, m, temp)
                    samples.append(sample)
                    samples_s.append(sample_s)
                samples = np.asarray(samples)  # 10x4x320x320
                samples_s = np.asarray(samples_s)  # 10x4xcx320x320
                samples = np.moveaxis(samples, 1, 0)  # 4x10x320x320
                samples_s = np.moveaxis(samples_s, 1, 0)  # 4x10xcx320x320
            else:
                samples, samples_s = predict(images, n_samples, model, temp)
            # calculate GT dice
            gt_dice.append(np.asarray(dice_coefficient(labels, samples)))

            # calculate predicted dice
            mean_dice_batch, var_dice_batch = predict_dice(samples_s)
            pred_dice.append(np.asarray(mean_dice_batch))
            pred_var.append(np.asarray(var_dice_batch))

        del images, samples, samples_s, labels
        gc.collect()

    gt_dice = np.concatenate(gt_dice)
    pred_var = np.concatenate(pred_var)
    pred_dice = np.concatenate(pred_dice)

    # ------------- get score function and qhat -------------
    scores = list(map(lambda y, f, u: abs((y - f)) / u, gt_dice, pred_dice, pred_var))
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)

    # ------------- apply to test set to get predicted performance range -------------
    gt_dice = []
    pred_dice = []
    pred_var = []
    for images, labels in test_loader:
        labels = labels.detach().cpu().numpy()
        if len(checkpoint_dir) > 1:
            ensemble = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
            samples = []
            samples_s = []
            for temp, m in zip(temps, ensemble):
                sample, sample_s = predict(images, n_samples, m, temp)
                samples.append(sample)
                samples_s.append(sample_s)
            samples = np.asarray(samples)  # 10x4x320x320
            samples_s = np.asarray(samples_s)  # 10x4xcx320x320
            samples = np.moveaxis(samples, 1, 0)  # 4x10x320x320
            samples_s = np.moveaxis(samples_s, 1, 0)  # 4x10xcx320x320
        else:
            samples, samples_s = predict(images, n_samples, model, temp)

        # calculate GT dice
        gt_dice.append(np.asarray(dice_coefficient(labels, samples)))

        # calculate predicted dice
        mean_dice_batch, var_dice_batch = predict_dice(samples_s)
        pred_dice.append(np.asarray(mean_dice_batch))
        pred_var.append(np.asarray(var_dice_batch))

        del images, samples, samples_s, labels
        gc.collect()

    gt_dice = np.concatenate(gt_dice)
    pred_var = np.concatenate(pred_var)
    pred_dice = np.concatenate(pred_dice)

    # save stuff
    np.savez("performanceprediction" + saving_name, gt_dice=gt_dice, pred_dice=pred_dice, pred_var=pred_var, qhat=qhat)

    ### ------------- plot overview -------------
    sorted_gt_dice = np.sort(gt_dice)
    ind_sorted_gt_dice = np.argsort(gt_dice)
    sorted_pred_dice = pred_dice[ind_sorted_gt_dice]
    sorted_pred_var = pred_var[ind_sorted_gt_dice]
    lower = sorted_pred_dice - qhat * sorted_pred_var
    upper = np.asarray(sorted_pred_dice + qhat * sorted_pred_var)
    upper[upper > 1] = 1
    lower[lower < 0] = 0

    x_values = np.asarray(range(len(sorted_gt_dice)))


    def plot_overview(title, ax):
        ax.fill_between(x_values, lower, upper,
                        color='silver', alpha=0.5, lw=0, label="Pred. range")
        sns.lineplot(x=x_values, y=sorted_gt_dice,
                     color="black", ax=ax, marker="", linestyle='-',
                     markeredgewidth=0, markersize=4, label="GT DSC")

        sns.lineplot(x=x_values, y=sorted_pred_dice,
                     ax=ax, marker=".", linestyle='',
                     markeredgewidth=0, markersize=4
                     )

        # Add labels
        ax.set_xlabel('Image index')
        ax.set_ylabel('Predicted Dice')
        ax.set_title(title)

        return ax


    with plt.style.context("./plot_style.txt"):
        fig, ax = plt.subplots(figsize=(2.2, 1.9), layout="constrained")
        ax = plot_overview("PHiSeg", ax)

        # Style the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='lower right')
        ax.get_legend().get_frame().set_linewidth(0.0)

        fig.savefig("PerformancePrediction" + saving_name + ".png")

    # ------------- print numerical results -------------
    def interval_percentage(lower_bounds, upper_bounds, ys):
        count_within_interval = 0
        for lower, upper, y in zip(lower_bounds, upper_bounds, ys):
            if lower <= y <= upper:
                count_within_interval += 1

        percentage = (count_within_interval / len(ys)) * 100
        return percentage


    print("Percentage of gt dice scores in intervall: ",
          interval_percentage(pred_dice - qhat * pred_var, pred_dice + qhat * pred_var, gt_dice))
    print("Mean width of bound: ", 2 * qhat * np.mean(pred_var))

    MSE = np.square(np.subtract(gt_dice, pred_dice)).mean()
    print("MSE: ", MSE)