import gc
import pandas as pd
from src.models import UNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from torchvision.transforms import v2
import random
from PIL import Image
import h5py

def apply_augmentation(image):
    # apply random augmentation
    # return transformed image and segmentation
    type = random.randint(0,8)
    angle = random.randrange(0, 360)

    image = Image.fromarray(image)

    if type == 0:
        transform = v2.ColorJitter(brightness=(0.5, 1.5))
        transformed = transform(image)
    elif type == 1:
        transform = v2.ColorJitter(hue=(0, 0.1))
        transformed = transform(image)
    elif type == 2:
        transform = v2.ColorJitter(saturation=(0.3, 1.5))
        transformed = transform(image)
    elif type == 3:
        transform = v2.ColorJitter(contrast=(0.4, 2))
        transformed = transform(image)
    elif type == 4:
        transformed = v2.functional.horizontal_flip(image)
    elif type == 5:
        transformed = v2.functional.vertical_flip(image)
    elif type == 6:
        transformed = v2.functional.rotate(image,angle)
        image = np.asarray(transformed).copy()
        return image, type, angle
    else:
        transform = v2.GaussianBlur(kernel_size=(3, 3), sigma=(1., 9.))
        transformed = transform(image)

    image = np.asarray(transformed).copy()

    return image, type

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient for a given pair of ground truth and predicted segmentation masks. Mean over labels.
    Parameters:
    - y_true (numpy.ndarray): Ground truth segmentation mask.
    - y_pred (numpy.ndarray): Predicted segmentation mask.
    Returns:
    - float: The Dice coefficient.
    """
    dice = 0
    for value in [1]:
        true_binary = y_true == value
        pred_binary = y_pred == value
        if true_binary.sum() + pred_binary.sum() == 0:
            dice += 1
            continue
        intersection = np.logical_and(true_binary, pred_binary)
        dice += (2 * intersection.sum()) / (true_binary.sum() + pred_binary.sum())
    return dice


def predict_dice(mat):
    """
        Compute the predicted Dice coefficient for a given predicted segmentation mask by calculating a confusion
        matrix.
        Parameters:
        - mat matrix of predicted segmentation
        Returns:
        - float: The Dice coefficient.
        """
    positives = mat > 0.5
    negatives = mat <= 0.5
    TP = mat[positives].sum()
    FP = positives.sum() - TP
    FN = mat[negatives].sum()
    # TN = negatives.sum() - FN
    return (2 * TP) / (2 * TP + FP + FN)


def get_samples(model, input, temp):
    # normalise image
    input = (input - input.mean(axis=(0, 1))) / input.std(axis=(0, 1))

    # change shape from (size, size, 3) to (3, size, size)
    input = np.moveaxis(input, -1, 0)

    # Convert to torch tensor
    input = torch.from_numpy(input)

    # Convert uint8 to float tensors
    input = input.type(torch.FloatTensor)
    input = torch.unsqueeze(input, 0)
    model_pred = model.predict(input)
    # temperature scale
    model.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1) * temp)
    model_pred = model.temperature_scaling(model_pred)
    s = torch.nn.Softmax(dim=1)
    samples_s = s(model_pred)  # 1x2x320x320
    samples_s = samples_s[0].detach().cpu().numpy()  # 2x320x320
    samples_s = samples_s[1]  # 320x320 only use softmax for label 1
    samples = torch.argmax(model_pred, dim=1)  # 1x320x320
    samples = samples.detach().cpu().numpy()
    samples = samples[0]  # remove batch dimension -> 320x320
    samples = samples.astype("uint8")
    return samples, samples_s

def predict(image, model, temp, t):
    """
        Predict n samples for a given sample
        Parameters:
        - image Image for which to predict
        - model Model that predicts image
        - temp Actual temp. scaling
        - tta If test time augmentation should be applied
        Returns:
        - samples hardened samples in shape 320 x 320
        - samples_s softmax samples in shape 2 x 320 x 320
        """
    samples_list = []
    samples_s_list = []
    # apply augmentations t times
    for i in range(t):
        aug_output = apply_augmentation(image)
        input = aug_output[0]
        type = aug_output[1]
        # predict for each augmentation
        samples, samples_s = get_samples(model, input, temp)
        # de-augment if needed
        if type == 4:
            samples_s = torch.from_numpy(samples_s)
            samples_s = v2.functional.horizontal_flip(samples_s)
            samples_s = samples_s.numpy()
            samples = torch.from_numpy(samples)
            samples = v2.functional.horizontal_flip(samples)
            samples = samples.numpy()
        elif type == 5:
            samples_s = torch.from_numpy(samples_s)
            samples_s = v2.functional.vertical_flip(samples_s)
            samples_s = samples_s.numpy()
            samples = torch.from_numpy(samples)
            samples = v2.functional.vertical_flip(samples)
            samples = samples.numpy()
        elif type == 6:
            samples_s = torch.from_numpy(samples_s)
            samples_s = torch.unsqueeze(samples_s, 0)
            samples_s = v2.functional.rotate(samples_s, -aug_output[2])
            samples_s = torch.squeeze(samples_s)
            samples_s = samples_s.numpy()
            samples = torch.from_numpy(samples)
            samples = torch.unsqueeze(samples, 0)
            samples = v2.functional.rotate(samples, -aug_output[2])
            samples = torch.squeeze(samples)
            samples = samples.numpy()
        samples_list.append(samples)
        samples_s_list.append(samples_s)
    return np.asarray(samples_list), np.asarray(samples_s_list)




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
        "--device",
        type=str,
        help="Either 'cpu' or 'gpu'",
        default='cpu',
        required=True
    )
    parser.add_argument(
        "--tta",
        type=str,
        help="Test Time Augmentation. Either 'True' or 'False'",
        default=False,
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of calibration epochs",
        default=1
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Saving folder",
        default='/'
    )
    parser.add_argument(
        "--augmentation",
        type=bool,
        help="whether to use augmentation during calibration",
        default=False
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoints_dir
    device = args.device
    folder = args.folder
    augmentation = args.augmentation
    tta = args.tta

    # data
    data_path = "/mnt/qb/work/baumgartner/bkc562/MasterThesis/FIVES_experiment.h5"
    hf = h5py.File(data_path, "r")
    cal_image = hf["cal"]["images"]
    cal_segmentation = hf["cal"]["label"]
    test_image = hf["test"]["images"]
    test_segmentation = hf["test"]["label"]
    test_id = hf["test"]["id"]

    # model
    model = UNet.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
    model.eval()
    saving_name = "ttaFIVES"

    n = len(cal_image)  # number of images in calibration set
    alpha = 0.1  # conformal prediction
    temp = 0.927
    t = 20  # number of augmentations tta
    epochs = args.epochs

    # ------------- get qhat scores -------------
    gt_dice = []
    pred_dice = []
    pred_var = []

    for epoch in range(epochs):
        for i in range(n):
            image = cal_image[i]
            label = cal_segmentation[i]
            samples, samples_s = predict(image, model, temp, t)
            # calculate GT dice
            result = np.asarray(list(map(lambda x: dice_coefficient(label, x), samples)))
            gt_dice.append(np.mean(result))
            # calculate predicted dice
            result = np.asarray(list(map(predict_dice, samples_s)))
            pred_dice.append(np.mean(result))
            pred_var.append(np.var(result))
            del image, samples, samples_s
            gc.collect()

    pred_var = np.asarray(pred_var).flatten()
    pred_dice = np.asarray(pred_dice).flatten()
    gt_dice = np.asarray(gt_dice).flatten()

    # ------------- get score function and qhat -------------
    scores = list(map(lambda y, f, u: abs((y - f)) / u, gt_dice, pred_dice, pred_var))
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)

    # ------------- apply to test set to get predicted performance range -------------
    gt_dice = []
    pred_dice = []
    pred_var = []

    for i in range(len(test_image)):
        image = test_image[i]
        label = test_segmentation[i]
        samples, samples_s = predict(image, model, temp, t)
        # calculate GT dice
        result = np.asarray(list(map(lambda x: dice_coefficient(label, x), samples)))
        gt_dice.append(np.mean(result))
        # calculate predicted dice
        result = np.asarray(list(map(predict_dice, samples_s)))
        pred_dice.append(np.mean(result))
        pred_var.append(np.var(result))
        del image, samples, samples_s
        gc.collect()

    pred_var = np.asarray(pred_var).flatten()
    pred_dice = np.asarray(pred_dice).flatten()
    gt_dice = np.asarray(gt_dice).flatten()

    # save stuff
    np.savez(folder + "performanceprediction" + saving_name, gt_dice=gt_dice, pred_dice=pred_dice, pred_var=pred_var, qhat=qhat)

    # ------------- data -------------
    sorted_gt_dice = np.sort(gt_dice)
    ind_sorted_gt_dice = np.argsort(gt_dice)
    sorted_pred_dice = pred_dice[ind_sorted_gt_dice]
    sorted_pred_dice = pred_dice[ind_sorted_gt_dice]
    sorted_pred_var = pred_var[ind_sorted_gt_dice]

    lower = sorted_pred_dice - qhat * sorted_pred_var
    upper = np.asarray(sorted_pred_dice + qhat * sorted_pred_var)
    upper[upper > 1] = 1
    lower[lower < 0] = 0
    x_values = np.asarray(range(len(sorted_gt_dice)))

    # label quality; quality = 1 for good quality; quality = 0 for bad quality
    # (>=2 of 3 quality features are marked as 0)
    quality_file = pd.read_excel("/mnt/qb/baumgartner/rawdata/FIVES/Quality Assessment.xlsx", sheet_name="Test")
    # Count the number of 0's in each row for the specified columns
    count_zeros = (quality_file[['IC', 'Blur', 'LC']] == 0).sum(axis=1)

    # Set 'quality' to 0 if at least one element are 0, otherwise set to 1
    quality_file["quality"] = (count_zeros < 1).astype(int)

    # sort quality labels in dataframe by indices in H5 file
    quality_sorted = np.zeros((160))
    for i in range(160):
        byte_str = test_id[i]

        decoded_str = byte_str.decode('utf-8')

        # extract the numeric part from the string
        numeric_part = ''.join(filter(str.isdigit, decoded_str))

        # convert the numeric part to an integer
        result = int(numeric_part)

        quality_sorted[i] = quality_file["quality"][result - 1]

    quality_sorted = quality_sorted[ind_sorted_gt_dice]
    # Create masks for marker 0 and marker 1
    mask_0 = quality_sorted == 0
    mask_1 = quality_sorted == 1

    # Apply masks to x_values and sorted_gt_dice
    x_values_1 = x_values[mask_1]
    x_values_0 = x_values[mask_0]
    sorted_gt_dice_1 = sorted_gt_dice[mask_1]
    sorted_gt_dice_0 = sorted_gt_dice[mask_0]

    # ------------- plot overview -------------
    fig, ax = plt.subplots()
    ax.fill_between(x_values, lower, upper,
                    color='gray', alpha=0.3)
    ax.scatter(x_values, sorted_pred_dice, label='Estimated DSC', s=2)
    ax.scatter(x_values_0, sorted_gt_dice_0, color='red', label='True DSC - bad quality', s=2, marker="o")
    ax.scatter(x_values_1, sorted_gt_dice_1, color='green', label='True DSC - good quality', s=2, marker="X")
    # Add legend
    plt.legend(loc='lower right')

    # Add labels
    plt.xlabel('image index')
    plt.ylabel('DSC')

    fig.savefig(folder + "PerformancePrediction" + saving_name + ".png")


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
