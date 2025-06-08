import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate_filter(kernel, angle):
    """Rotate a filter kernel by angle degrees."""
    size = kernel.shape[0]
    center = (size // 2, size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(kernel, rot_mat, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def log_filter(size, sigma):
    """Laplacian of Gaussian (LoG)."""
    half = size // 2
    x, y = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1))
    norm = (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
    gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    log = norm * gauss
    return log - np.mean(log)

def gaussian_2d(size, sigma_x, sigma_y, order_x=0, order_y=0):
    """2D Gaussian and its derivatives."""
    half = size // 2
    x, y = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1))
    gauss = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    if order_x == 1:
        gauss *= -x / (sigma_x**2)
    elif order_x == 2:
        gauss *= (x**2 - sigma_x**2) / (sigma_x**4)

    if order_y == 1:
        gauss *= -y / (sigma_y**2)
    elif order_y == 2:
        gauss *= (y**2 - sigma_y**2) / (sigma_y**4)

    return gauss - np.mean(gauss)

def create_lm_filter_bank(version='LMS'):
    """
    Create the Leung-Malik (LM) filter bank with 48 filters.
    version: 'LMS' (small) or 'LML' (large)
    Returns: list of 48 filters (each as a 2D NumPy array)
    """
    if version == 'LMS':
        base_scales = [1.0, np.sqrt(2), 2.0, 2*np.sqrt(2)]
    elif version == 'LML':
        base_scales = [np.sqrt(2), 2.0, 2*np.sqrt(2), 4.0]
    else:
        raise ValueError("Version must be 'LMS' or 'LML'.")

    orientations = [0, 30, 60, 90, 120, 150]
    elongation = 3.0
    size = 49  # odd number recommended
    filters = []

    # First and second Gaussian derivatives: 18 + 18 = 36 filters
    for scale in base_scales[:3]:  # only first 3 scales
        sigma_x = scale
        sigma_y = elongation * scale
        for angle in orientations:
            # First derivative (∂G/∂x)
            f1 = gaussian_2d(size, sigma_x, sigma_y, order_x=1, order_y=0)
            filters.append(rotate_filter(f1, angle))

            # Second derivative (∂²G/∂x²)
            f2 = gaussian_2d(size, sigma_x, sigma_y, order_x=2, order_y=0)
            filters.append(rotate_filter(f2, angle))

    # LoG filters: 8 total
    for sigma in [base_scales[0], 3 * base_scales[0]]:
        for _ in range(4):  # 4 versions at each scale
            filters.append(log_filter(size, sigma))

    # Gaussian filters (4)
    for sigma in base_scales:
        filters.append(gaussian_2d(size, sigma, sigma))

    return filters

def plot_filter_bank(filters, title="Filter Bank", ncols=8):
    """
    Plot a filter bank in a grid.
    filters: list of 2D NumPy arrays
    title: plot title
    ncols: number of columns in grid
    """
    n_filters = len(filters)
    nrows = (n_filters + ncols - 1) // ncols  # ceil division

    fig, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # If axs is 1D, flatten; else flatten 2D grid
    axs = np.array(axs).reshape(-1)

    for i, ax in enumerate(axs):
        if i < n_filters:
            ax.imshow(filters[i], cmap='gray')
        ax.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    LMS_filters = create_lm_filter_bank()
    LML_filters = create_lm_filter_bank(version="LML")
    plot_filter_bank(LMS_filters)
    plot_filter_bank(LML_filters)