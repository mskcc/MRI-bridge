import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.io as sio
from pathlib import Path
from scipy import ndimage
import scipy.stats as stats
from scipy.optimize import brentq
from preprocessing.split_config_demo import CAL_EXAMS
# ============================================================
# CALIBRATION: adapted from https://github.com/aangelopoulos/im2im-uq
# ============================================================
def _fraction_missed_loss(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Returns one miss fraction per image withing region of interest (target !=0).
    """
    if lower.shape != upper.shape or lower.shape != target.shape:
        raise ValueError(
            f"lower, upper, target must match. Got {lower.shape}, "
            f"{upper.shape}, {target.shape}."
        )

    misses = (lower[target!=0] > target[target!=0]).float()
    misses += (upper[target!=0] < target[target!=0]).float()
    misses = torch.clamp(misses, 0.0, 1.0)
    return misses.tolist(), torch.sum(target!=0).item()


def _apply_lambda(
    recon: torch.Tensor,
    lower_pred: torch.Tensor,
    upper_pred: torch.Tensor,
    lam: float,
):

    lower_cal = torch.clamp(recon - lam * (recon - lower_pred), min=0.0)
    upper_cal = recon + lam * (upper_pred - recon)

    return lower_cal, upper_cal


def _h1(y, mu):
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


def _hoeffding_plus(mu, x, n):
    return -n * _h1(np.minimum(mu, x), mu)


def _bentkus_plus(mu, x, n):
    return np.log(max(stats.binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1


def _HB_mu_plus(muhat, n, delta, maxiters=1000):
    def _tailprob(mu):
        hoeffding_mu = _hoeffding_plus(mu, muhat, n)
        bentkus_mu = _bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1.0

    try:
        return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)
    except Exception:
        return 1.0


# ============================================================
# calibration settings
# ============================================================

calibration_delta = 0.1
calibration_lambdas_start = 1.2
calibration_lambdas_end = 2.2
calibration_lambdas_steps = 100

PRINT_HISTO = True

lambdas = torch.linspace(
    calibration_lambdas_start,
    calibration_lambdas_end,
    steps = calibration_lambdas_steps,
    )
dlambda = lambdas[1] - lambdas[0]
alpha = 0.1

# Replace with your actual directory path
RESULT_FOLDER = os.environ["DEMO_PATH"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
RUN_NAME = os.environ["RUN_NAME"]
PROVIDED_RESULTS = os.environ["PROVIDED_RESULTS"]
target_dir = Path(RESULT_FOLDER + PROVIDED_RESULTS + '/PURE_Results')
mask_dir =  Path(RESULT_FOLDER + PROVIDED_RESULTS + '/Segmentations/')


# Find all calibration files
all_files = []
# .rglob("*") finds all files and folders recursively
for item in target_dir.rglob("*"):
    if item.is_file() and item.stem.split('_')[0] in CAL_EXAMS:
        all_files.append(item)

all_files = sorted(all_files)
n_total = len(all_files)

filename_current = 'Dummy'
flat_data_all = []
file_id = 0

# Iterate through files, load them along with segmentation masks, and combine the files
for index, file in enumerate(all_files):

    data = sio.loadmat(file)
    SR_images = torch.tensor(data["SR_images_corrected"])
    ref = torch.tensor(data["ref_corrected"])

    if index == 0:
        SR_images_all = torch.zeros((n_total, *SR_images.shape)) 
        ref_all = torch.zeros((n_total, *ref.shape)) 
        masks_all = torch.zeros((n_total, *ref.shape)) 
        combined_mask_all = torch.zeros((n_total, *ref.shape))

    SR_images_all[index] = SR_images
    ref_all[index] = ref

    filename = all_files[index].name.split('_')[0]
    if filename != filename_current:
        file_id += 1
        filename_current = filename
        segmentation_mask = np.load(f'{mask_dir}/Seg_{filename_current}.npz')
        mask = segmentation_mask['mask']

        if file_id == 1:
            masks_all = torch.tensor(mask)
        else:
            masks_all = torch.cat([masks_all, mask], dim=0) 

if len(masks_all.shape) == 2:
    masks_all = masks_all[None, ...]

if PRINT_HISTO is True:
    # Z-score normalize the entire array
    mean_vals = torch.mean(SR_images_all, axis=(1))
    std_vals = torch.std(SR_images_all, axis=(1))
    normalized_data = (SR_images_all - mean_vals[:, None, ...]) / std_vals[:, None, ...]
    # Extract non-zero values and flatten the array
    valid_data = torch.masked_select(normalized_data, masks_all[:,None,...].to(torch.bool))
    flat_data = valid_data.flatten()

    mu = 0       # Mean (center of the curve)
    sigma = 1    # Standard deviation (width of the curve)

    # 2. Generate x values (spanning 4 standard deviations left and right)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

    # 3. Calculate the Gaussian probability density function (PDF)
    y = stats.norm.pdf(x, mu, sigma)

    # 4. Create and plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_data, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.plot(x, y, label=f'Gaussian ($\mu$={mu}, $\sigma$={sigma})', color='royalblue', linewidth=2)
    plt.title('Histogram of Normalized Non-Zero Values (0th Dimension)')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(Path(RESULT_FOLDER + EXPERIMENT_NAME + f'/{RUN_NAME}_histo.png'))
    plt.close()

    # --- Calculate Tails (Kurtosis) ---
    scipy_skew = stats.skew(flat_data)
    scipy_kurt = stats.kurtosis(flat_data)
    print(f"Symmetry (Skewness): {scipy_skew:.4f}")
    print(f"Tail Weight (Excess Kurtosis): {scipy_kurt:.4f}")

best_lambda = None

for lam in reversed(lambdas):
    cal_losses = []
    cal_N = []

    for test_data_idx in range(n_total):

        lower_pred, upper_pred = torch.mean(SR_images_all[test_data_idx], dim=0) - 1.645 * torch.std(SR_images_all[test_data_idx], dim=0), \
            torch.mean(SR_images_all[test_data_idx], dim=0) + 1.645 * torch.std(SR_images_all[test_data_idx], dim=0)
        
        lower_cal, upper_cal = _apply_lambda(
            torch.mean(SR_images_all[test_data_idx], dim=0) * masks_all[test_data_idx], lower_pred * masks_all[test_data_idx], upper_pred * masks_all[test_data_idx], lam-dlambda)

        loss, N = _fraction_missed_loss(lower_cal, upper_cal, ref_all[test_data_idx] * masks_all[test_data_idx])
        cal_losses.extend(loss)
        cal_N.append(N)

    Rhat = np.mean(cal_losses)
    print(f'lamda is {lam} and Rhat is {Rhat}')
    RhatPlus = _HB_mu_plus(Rhat, sum(cal_N), calibration_delta)

    print(
        f"[Calibration] lambda={lam.item():.4f} "
        f"Rhat={Rhat:.4f} RhatPlus={RhatPlus:.4f}"
    )

    if Rhat >= alpha or RhatPlus > alpha:
        best_lambda = float(lam.item())
        print(f"[Final Calibration] lambda={best_lambda:.4f} ")
        break

if best_lambda is None:
    raise RuntimeError("Optimal calibration lambda not found in scanned range.")
