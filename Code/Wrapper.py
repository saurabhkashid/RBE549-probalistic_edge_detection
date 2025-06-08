#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

def gaussian_kernel(size, sigma):
    """Create 2D Gaussian kernel manually with NumPy."""
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def sobel_kernel_x():
    """Classic Sobel X kernel. This is derivative/grdient filter """
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

def rotate_filter(kernel, angle):
    """Rotate a filter kernel by angle degrees."""
    size = kernel.shape[0]
    center = (size // 2, size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(kernel, rot_mat, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def convolve2d(image, kernel):
    """Manual convolution with valid padding."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Use OpenCV to pad the image
    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REPLICATE)
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def create_dog_filter_bank(scales, num_orientations, kernel_size=31):
    """ Difference of Gaussian (DOG)"""
    filter_bank = []
    sobel = sobel_kernel_x()

    for sigma in scales:
        gauss = gaussian_kernel(kernel_size, sigma)
        dog = convolve2d(gauss, sobel)  # sobel is linear filter so cross-corelation and covolution is similar
        for angle in np.linspace(0, 360, num_orientations, endpoint=False):
            rotated = rotate_filter(dog, angle)
            filter_bank.append(rotated)
    return (filter_bank, len(scales), num_orientations)

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

def plot_filters(filter_type):
    generators = {
        'dog': lambda: create_dog_filter_bank(scales=[1.0, 2.0], num_orientations=16),
        'lms': lambda: create_lm_filter_bank('LMS'),
        'lml': lambda: create_lm_filter_bank('LML')
    }

    if filter_type not in generators:
        print(f"Unknown filter type: {filter_type}")
        return

    filters = generators[filter_type]()
    if isinstance(filters, tuple):  # e.g., DoG returns (filters, rows, cols)
        filters, rows, cols = filters
    else:
        rows, cols = 6, 8  # LM filters are always 48 = 6x8

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            axs[i, j].imshow(filters[idx], cmap='gray')
            axs[i, j].axis('off')
            if i == 0 and filter_type == 'dog':
                axs[i, j].set_title(f"{int(j * 360 / cols)}°", fontsize=8)

    plt.suptitle(f"{filter_type.upper()} Filter Bank", fontsize=16)
    plt.tight_layout()
    plt.show()


def gabor_filter(size, wavelength, theta, sigma, gamma=0.5, psi=0, scales = [1,2,3,4]):
	"""
	Generate a Gabor filter using the given parameters.
	:param size: Kernel size (odd int)
	:param wavelength: Wavelength of the sinusoid (λ)
	:param theta: Orientation in degrees
	:param sigma: Standard deviation of Gaussian
	:param gamma: Aspect ratio (default 0.5)
	:param psi: Phase offset (default 0)
	:return: Gabor filter (2D numpy array)
	"""
	theta = np.deg2rad(theta)
	half = size // 2
	x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))

	# Rotate coordinates
	x_prime = x * np.cos(theta) + y * np.sin(theta)
	y_prime = -x * np.sin(theta) + y * np.cos(theta)

	# Gabor formula
	gaussian = np.exp(-(x_prime**2 + (gamma**2) * y_prime**2) / (2 * sigma**2))
	sinusoid = np.cos(2 * np.pi * x_prime / wavelength + psi)

	return gaussian * sinusoid


def create_gabor_filter_bank(scales, orientations, size=31):
    """
    Create a Gabor filter bank with multiple scales and orientations.
    :param scales: list of (wavelength, sigma) pairs
    :param orientations: list of angles in degrees
    :param size: filter size
    :return: list of Gabor filters
    """
    filters = []
    for wavelength, sigma in scales:
        for theta in orientations:
            kernel = gabor_filter(size, wavelength, theta, sigma)
            filters.append(kernel)
    return filters

def plot_gabor_filter_bank():

	scales = [
		(4, 2.0),
		(8, 4.0),
		(16, 6.0)
	]

	# Define orientations (in degrees)
	orientations = [0, 30, 60, 90, 120, 150]
	num_scales=len(scales)
	num_orientations=len(orientations)
	filters = create_gabor_filter_bank(scales, orientations)
	fig, axs = plt.subplots(num_scales, num_orientations, figsize=(2*num_orientations, 2*num_scales))
	for i in range(num_scales):
		for j in range(num_orientations):
			idx = i * num_orientations + j
			axs[i, j].imshow(filters[idx], cmap='gray')
			axs[i, j].axis('off')
			axs[i, j].set_title(f"θ={j*180//num_orientations}°, s={i}", fontsize=8)
	plt.suptitle("Gabor Filter Bank (Multi-scale, Multi-orientation)", fontsize=16)
	plt.tight_layout()
	plt.show()

def plot_all_lm_filters(version='LMS'):
    """
    Plot all 48 Leung-Malik filters in a 6x8 grid.
    :param version: 'LMS' or 'LML'
    """
    filters = create_lm_filter_bank(version)
    rows, cols = 6, 8  # 48 filters
    assert len(filters) == 48, f"Expected 48 filters, got {len(filters)}"

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for idx, f in enumerate(filters):
        i, j = divmod(idx, cols)
        axs[i, j].imshow(f, cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(f"#{idx}", fontsize=8)

    plt.suptitle(f"Leung-Malik Filter Bank ({version})", fontsize=16)
    plt.tight_layout()
    plt.show()

def compute_filter_responses(image_gray, filter_bank):
    """
    Apply each filter to the grayscale image and stack the responses.
    :param image_gray: (H, W) grayscale image
    :param filter_bank: list of 2D filters
    :return: (H, W, N) response volume
    """
    H, W = image_gray.shape
    N = len(filter_bank)
    responses = np.zeros((H, W, N), dtype=np.float32)
    for i, f in enumerate(filter_bank):
        responses[:, :, i] = convolve(image_gray, f, mode='reflect')
    return responses

def compute_texton_map(image_gray, filter_bank, K=64):
    """
    Cluster filter responses at each pixel to get a texton map.
    :return: (H, W) texton label map
    """
    responses = compute_filter_responses(image_gray, filter_bank)
    H, W, N = responses.shape
    response_vectors = responses.reshape(-1, N)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(response_vectors)
    texton_labels = kmeans.labels_.reshape(H, W)

    return texton_labels

def compute_brightness_map(image_gray, K=16):
    """
    Cluster grayscale intensities into brightness bins.
    :return: (H, W) brightness label map
    """
    H, W = image_gray.shape
    pixels = image_gray.reshape(-1, 1)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    brightness_labels = kmeans.labels_.reshape(H, W)
    return brightness_labels

def compute_color_map(image_rgb, K=16, use_lab=True):
    """
    Cluster RGB (or Lab) values using KMeans.
    :return: (H, W) color label map
    """
    H, W, _ = image_rgb.shape
    if use_lab:
        image_lab = rgb2lab(image_rgb)
        pixels = image_lab.reshape(-1, 3)
    else:
        pixels = image_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    color_labels = kmeans.labels_.reshape(H, W)
    return color_labels

def show_label_map(label_map, title="Label Map"):
    plt.imshow(label_map, cmap='tab20')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def create_half_disc_mask(orientations=8, radii=[4, 10, 20]):
    """
    Generate half-disc masks oriented along true dividing lines (not always horizontal).
    """
    masks = []
    for r in radii:
        size = 2 * r + 1
        center = r
        y, x = np.ogrid[:size, :size]
        circle = (x - center)**2 + (y - center)**2 <= r**2

        for i in range(orientations):
            angle = i * (180 / orientations)
            theta = np.deg2rad(angle)

            # Direction vector for splitting line
            nx, ny = np.cos(theta), np.sin(theta)
            # Signed distance from center line
            distance = (x - center) * nx + (y - center) * ny

            half1 = (distance > 0) & circle
            half2 = (distance <= 0) & circle

            masks.append((half1.astype(np.uint8), half2.astype(np.uint8)))

    return masks

def plot_half_disc(masks):
    # Show the 8 pairs
    fig, axs = plt.subplots(2, 24, figsize=(16, 4))
    for i, (m1, m2) in enumerate(masks):
        axs[0, i].imshow(m1, cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f'Half 1\n{int(i * 180 / 8)}°')
        axs[1, i].imshow(m2, cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Half 2\n{int(i * 180 / 8)}°')

    plt.suptitle("Half-Disc Mask Pairs", fontsize=16)
    plt.tight_layout()
    plt.show(block=True)

def compute_chi_square_gradient(label_map, masks, num_bins):
    """
    Compute chi-square gradient map for a label map using a set of half-disc masks.
    
    :param label_map: (H, W) image of clustered labels (e.g., Texton map, Brightness map, Color map)
    :param masks: list of (left_mask, right_mask) pairs (each a binary 2D array)
    :param num_bins: number of clusters (K)
    :return: (H, W, N) gradient magnitude array, where N = number of mask pairs
    """
    H, W = label_map.shape
    N = len(masks)
    gradients = np.zeros((H, W, N), dtype=np.float32)
    epsilon = 1e-10

    for k in range(num_bins):
        binary_mask = (label_map == k).astype(np.float32)
        for i, (left_mask, right_mask) in enumerate(masks):
            g_i = convolve(binary_mask, left_mask, mode='reflect')
            h_i = convolve(binary_mask, right_mask, mode='reflect')
            num = (g_i - h_i) ** 2
            denom = g_i + h_i + epsilon
            gradients[:, :, i] += 0.5 * num / denom  # accumulate chi-sqr term
    return gradients.max(axis=2)

def read_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # return unit8 dtype
        img = img.astype(np.float32)
        if img is not None:
            # normalize the image
            img_norm = cv2.normalize(img,  None, 0, 1, cv2.NORM_MINMAX)
            images.append(img_norm)

    return images

def compute_pb_edge(Tg, Bg, Cg, canny_pb, sobal_pb):
    w1  = 0.5
    w2 = 0.5
    # Mean feature strength at each pixel
    avg_feature = (Tg + Bg + Cg) / 3.0  # shape (H, W)

    # Combine Canny and Sobel
    baseline = w1 * canny_pb + w2 * sobal_pb  # shape (H, W)

    # Final Pb edge probability
    pb_edges = avg_feature * baseline
    pb_edges = cv2.normalize(pb_edges, None, 0, 1, norm_type=cv2.NORM_MINMAX)  # optional normalize to [0,1]
    return pb_edges

def main():

    # read the baseline images from the folder
    sobel_baseline_folder = "Phase1/BSDS500/SobelBaseline"
    cannys_baseline_folder = "Phase1/BSDS500/CannyBaseline"
    test_images_folder = "Phase1/BSDS500/Images"
    gt_folder = "Phase1/BSDS500/GroundTruth"
    folder_to_save = "pictures/result"

    sobel_baseline_imgs = read_images(sobel_baseline_folder)
    cannys_baseline_imgs = read_images(cannys_baseline_folder)
    test_imgs = read_images (test_images_folder)
    gt_imgs = read_images(gt_folder)

    for i in range(len(sobel_baseline_imgs)):
        canny = cannys_baseline_imgs[i]
        sobel = sobel_baseline_imgs[i]
        test_img_gry = test_imgs[i]
        test_img_rgb = cv2.cvtColor((test_img_gry * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Create a filter bank first (e.g., LMS)
        lm_filters = create_lm_filter_bank('LMS')

        # Compute maps
        lm_texton_map = compute_texton_map(test_img_gry, lm_filters, K=64)
        brightness_map = compute_brightness_map(test_img_gry, K=16)
        color_map = compute_color_map(test_img_rgb, K=16)

        #==== save the map ========
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        axs[0].imshow(lm_texton_map, cmap='tab20')
        axs[0].set_title('Texton-Map')
        axs[0].axis('off')

        axs[1].imshow(brightness_map, cmap='tab20')
        axs[1].set_title('Brightness-Map')
        axs[1].axis('off')

        axs[2].imshow(color_map, cmap='tab20')
        axs[2].set_title('Color-Map')
        axs[2].axis('off')

        # save texton- map 
        map_folder = "pictures/texton_map"
        filename = f"Map_of_image_{i}.png"
        save_path = os.path.join(map_folder, filename)
        plt.savefig(save_path)
        plt.close(fig)

        # Show
        # show_label_map(lm_texton_map, "Texton Map (T)")
        # show_label_map(brightness_map, "Brightness Map (B)")
        # show_label_map(color_map, "Color Map (C)")

        # plot half disc
        half_disc_masks = create_half_disc_mask() # 24 masks 8 orientation 3 redii
        plot_half_disc(half_disc_masks)

        # calculate chi square
        texton_gradients = compute_chi_square_gradient(lm_texton_map, half_disc_masks, num_bins=64)
        brightness_gradiants = compute_chi_square_gradient(brightness_map, half_disc_masks, 16)
        color_gradiants = compute_chi_square_gradient(color_map, half_disc_masks, 16)
        pb = compute_pb_edge(texton_gradients, brightness_gradiants, color_gradiants, canny, sobel )

        # ===============Plot and save result image===============
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        axs[0].imshow(sobel, cmap='gray')
        axs[0].set_title('Sobel Baseline')
        axs[0].axis('off')

        axs[1].imshow(canny, cmap='gray')
        axs[1].set_title('Canny Baseline')
        axs[1].axis('off')

        axs[2].imshow(pb, cmap='gray')
        axs[2].set_title('Pb-lite Result')
        axs[2].axis('off')

        axs[3].imshow(gt_imgs[i], cmap='gray')
        axs[3].set_title('Ground Truth')
        axs[3].axis('off')

        plt.tight_layout()

        # Save figure
        filename = f"max_result_image_{i}.png"
        save_path = os.path.join(folder_to_save, filename)
        plt.savefig(save_path)
        plt.close(fig)  # close the figure to save memory
        print("done")

if __name__ == '__main__':
    main()



