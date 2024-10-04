# 1) load cal and test data (now includes random ood images)
# 2) OOD detection: Liu, Weitang, et al. "Energy-based out-of-distribution detection."
#                   Advances in neural information processing systems 33 (2020): 21464-21475.
#       on cal data
#           fine-tune
#           compute ood scores using energy function
#           tune threshold for ood decision
#       on test
#           compute ood scores using energy function
#           detect ood images and sort out
# 3) Performance prediction:
#       on cal data
#           calibrate and compute scores
#       on test data
#           compute performance range
# 4) Image quality
#       on test data
#           sort out all images with quality score = 0 or ood = 1

from src.models import PHISeg
import numpy as np
import torch
from src.data import FIVES
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
import subprocess
import random
from torch.utils.data.dataset import Dataset
import h5py
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------
# 1) data
# ---------------------------------------------------------------------------------------------------------------------
class Fives_ood(Dataset):
    def __init__(self, file_path, distribution, t: str, transform=None):
        hf = h5py.File(file_path, "r")

        self.transform = transform

        if t in ["train", "val", "test", "cal"]:
            out_mask = hf[t]["ood"][:] == 1
            in_mask = hf[t]["ood"][:] == 0

            if distribution == "in":
                self.images = hf[t]["images"][in_mask]
                self.segmentations = hf[t]["label"][in_mask]
                self.id = hf[t]["id"][in_mask]
                self.ood = hf[t]["ood"][in_mask]
                self.quality = hf[t]["quality"][in_mask]
            elif distribution == "out":
                self.images = hf[t]["images"][out_mask]
                self.segmentations = hf[t]["label"][out_mask]
                self.id = hf[t]["id"][out_mask]
                self.ood = hf[t]["ood"][out_mask]
                self.quality = hf[t]["quality"][out_mask]
            else:
                raise ValueError(f"Unknown distribution specifier: {distribution}")
        else:
            raise ValueError(f"Unknown test/train/val/cal specifier: {t}")

    def __getitem__(self, index):

        image = self.images[index]
        segmentation = self.segmentations[index]

        if self.transform == None:
            pass
        else:
            if random.random() > 0.5:
                image, segmentation = self.transform(image, segmentation)

        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size)
        image = np.moveaxis(image, -1, 0)

        # Convert to torch tensor
        image = torch.from_numpy(image)
        segmentation = torch.from_numpy(segmentation)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        segmentation = segmentation.type(torch.LongTensor)

        return image, segmentation

    def __len__(self):
        return self.images.shape[0]

    def get_id(self, i):
        return self.id[i]

data_path = "/mnt/qb/work/baumgartner/bkc562/MasterThesis/FIVES_experiment.h5"
cal_dataset = FIVES.Fives(file_path=data_path, t="cal", transform=None)
test_dataset = FIVES.Fives(file_path=data_path, t="test", transform=None)
cal_loader = DataLoader(cal_dataset, batch_size=4, drop_last=False, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, drop_last=False, shuffle=False)

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
    dice_scores = []
    batch_dice_scores = []

    for i in range(batch_size):
        y_t = y_true[i]
        for j in range(n):
            y_p = y_pred[i, j]
            dice = 0
            for value in [0, 1]:
                true_binary = y_t == value
                pred_binary = y_p == value
                if true_binary.sum() + pred_binary.sum() == 0:
                    dice += 0
                else:
                    intersection = np.logical_and(true_binary, pred_binary)
                    dice += (2 * intersection.sum()) / (true_binary.sum() + pred_binary.sum())
            dice_scores.append(dice / 2)
        batch_dice_scores.append(np.mean(dice_scores))

    return batch_dice_scores

# ---------------------------------------------------------------------------------------------------------------------
# 2) OOD detection
# ---------------------------------------------------------------------------------------------------------------------
def finetune():
    train_in = Fives_ood(file_path=data_path, distribution="in", t="cal", transform=None)
    train_out = Fives_ood(file_path=data_path, distribution="out", t="cal", transform=None)
    train_loader_in = DataLoader(train_in, batch_size=2, drop_last=False, shuffle=False)
    train_loader_out = DataLoader(train_out, batch_size=2, drop_last=False, shuffle=False)

    optimizer = torch.optim.SGD(
        model.parameters(), 0.001, momentum=0.9,
        weight_decay=0.0005, nesterov=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            10 * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / 0.001))

    model.train()  # enter train mode

    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        # forward

        x = model.predict_output_samples(data, N=20)
        x = torch.mean(x, dim=1)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-(-25.)), 2).mean() + torch.pow(F.relu((-7.)-Ec_out), 2).mean())

        loss.backward()
        optimizer.step()


def get_ood_scores(loader):
    ood_score = []
    dice_score = []
    T = 1
    with torch.no_grad():
        for data, target in loader:
            x = model.predict_output_samples(data, N=2)  # 4x20x2x320x320
            x = torch.mean(x, dim=1)  # 4x2x320x320
            s = torch.nn.Softmax(dim=1)
            x = s(x)

            # ood score
            _score = - T * torch.logsumexp(x / T, dim=1)
            _score = torch.mean(_score, dim=(1, 2))
            ood_score.append(_score.numpy())

            # dice score
            dice_score.append(dice_coefficient(target, torch.argmax(x, dim=1)))

    return np.concatenate(ood_score, axis=0), np.concatenate(dice_score, axis=0)

def get_ood_threshold(loader):
    scores, _ = get_ood_scores(loader)
    scores = torch.from_numpy(scores)
    s = torch.nn.Softmax(dim=0)
    scores = s(scores)
    scores = scores.numpy().copy()
    ood_target = cal_dataset.ood[:]

    best_threshold = 0
    best_youden_index = 0

    # Youden's J statistic to get the best threshold
    for threshold in list(np.arange(0.1, 0.8, 0.1)):
        ood_pred = np.asarray([1 if score <= threshold else 0 for score in scores])
        conf = confusion_matrix(ood_target, ood_pred).ravel()
        print(conf)
        tn, fp, fn, tp = conf
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        youdens_ind = tpr - fpr
        if youdens_ind > best_youden_index:
            best_youden_index = youdens_ind
            best_threshold = threshold

    return best_threshold

def ood_performance(loader):
    scores, _ = get_ood_scores(loader)
    ood_target = test_dataset.ood
    return roc_auc_score(ood_target, scores)

model = PHISeg.load_from_checkpoint("/mnt/qb/work/baumgartner/bkc562/MasterThesis/runsFIVES/517aa3a-seed=0-phiseg--bs=4/version_0/checkpoints/best-dice-epoch=257-step=30870.ckpt", map_location=torch.device("cpu"))

'''
finetune()
best_threshold = get_ood_threshold(cal_loader)
ood_scores, DSC = get_ood_scores(test_loader)
s = torch.nn.Softmax(dim=0)
ood_scores = s(ood_scores)
ood_scores = ood_scores.numpy().copy()
id_pred = [False if score <= best_threshold else True for score in ood_scores]  # mask for id images
ood_pred = [True if score <= best_threshold else False for score in ood_scores]  # mask for ood images
in_DSC = DSC[id_pred]
ood_DSC = DSC[ood_pred]
below_DSC = in_DSC[in_DSC < 0.8]
print("Number of in distribution images with DSC below 0.8: ", below_DSC.shape[0], " and values: ", below_DSC)
print("Number of ood images: ", ood_DSC.shape[0])
print("OOD detection performance on test set: ", ood_performance(test_loader))

del model
'''
# ---------------------------------------------------------------------------------------------------------------------
# 1) Get GT Dice (and Performance Prediction values)
# ---------------------------------------------------------------------------------------------------------------------
def call_performance_prediction(checkpoints_dirs, n_samples, device):
    command = [
        "python", "PerformancePrediction.py",
        "--checkpoints_dir"
    ] + checkpoints_dirs + [
        "--n_samples", str(n_samples),
        "--device", device
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

checkpoints_dirs = ["/mnt/qb/work/baumgartner/bkc562/MasterThesis/runsFIVES/517aa3a-seed=0-phiseg--bs=4/version_0/checkpoints/best-dice-epoch=257-step=30870.ckpt"]
n_samples = 100
device = "cpu"
#call_performance_prediction(checkpoints_dirs, n_samples, device)

data_pp = np.load("performancepredictionphisegFIVES.npz")

pred_dice = data_pp["pred_dice"]
qhat = data_pp["qhat"]
pred_var = data_pp["pred_var"]
gt_dice = data_pp["gt_dice"]

lower = pred_dice - qhat * pred_var
upper = np.asarray(pred_dice + qhat * pred_var)
upper[upper > 1] = 1
lower[lower < 0] = 0

#outsort = gt_dice[lower < 0.8]
#insort = gt_dice[lower >= 0.8]

#failed = insort[insort < 0.8]

#print("Number of images with predicted DSC above 0.8 but GT DSC below 0.8: ", failed.shape[0], " and values: ", failed)
#print("Number of outsorted images: ", outsort.shape[0])
'''
# ---------------------------------------------------------------------------------------------------------------------
# 3) Image Quality
# ---------------------------------------------------------------------------------------------------------------------
outsort = gt_dice[(test_dataset.ood == 1) | (test_dataset.quality == 0)]
insort = gt_dice[(test_dataset.ood == 0) | (test_dataset.quality == 1)]

failed = insort[insort < 0.8]
print("Number of images with bad quality but DSC below 0.8: ", failed.shape[0], " and values: ", failed)
print("Number of outsorted images: ", outsort.shape[0])
'''

def get_curve(dice_list, order):
    curve = []
    # no element removed
    mean_dice = np.mean(dice_list)
    curve.append(mean_dice)

    # remove element from worst to best performance
    for i in range(len(order) - 1):
        dice_list = np.delete(dice_list, order[i])
        mean_dice = np.mean(dice_list)
        curve.append(mean_dice)
        order = [idx - 1 if idx > order[i] else idx for idx in order]

    # only last element left
    mean_dice = np.mean(dice_list)
    curve.append(mean_dice)

    return curve

dice_list = gt_dice.copy()

# ---------------------------------------------------------------------------------------------------------------------
# 2) Performance Prediction
# ---------------------------------------------------------------------------------------------------------------------
sorted_lower_idx = np.argsort(lower)
performanceprediction_curve = get_curve(dice_list, sorted_lower_idx)

# ---------------------------------------------------------------------------------------------------------------------
# 3) OOD
# ---------------------------------------------------------------------------------------------------------------------
sorted_ood_score = np.argsort(test_dataset.ood)
OOD_curve = get_curve(dice_list, sorted_ood_score)

# ---------------------------------------------------------------------------------------------------------------------
# 4) Quality
# ---------------------------------------------------------------------------------------------------------------------
sorted_quality_score = np.argsort(test_dataset.quality)
quality_curve = get_curve(dice_list, sorted_quality_score)

# ---------------------------------------------------------------------------------------------------------------------
# 5) Plot curve
# ---------------------------------------------------------------------------------------------------------------------
x = np.arange(161)
with plt.style.context("./plot_style.txt"):
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x, performanceprediction_curve, label="Performance prediction", linestyle='-', color='limegreen')
    ax.plot(x, OOD_curve, label="OOD detection", linestyle='--', color='darkorange')
    ax.plot(x, quality_curve, label="image quality control", linestyle='-.', color='steelblue')

    ax.set_ylabel('DSC')
    ax.set_xlabel('Number of images removed')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig("FIVES_experiment_pp_vs_upperbound.png")
