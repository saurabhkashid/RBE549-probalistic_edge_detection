import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def plot_DOG_filters():
    filters, rows, cols = create_dog_filter_bank(scales=[1.0, 2.0], num_orientations=16)
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            axs[i, j].imshow(filters[idx], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f"{int(j * 360 / cols)}°", fontsize=8)
    plt.suptitle(f"DOG Filter Bank", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_DOG_filters()