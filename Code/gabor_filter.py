import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def gabor_filter(size, wavelength, theta, sigma, gamma=0.95, psi=0, scales = [1,2,3,4]):
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
	sinusoid = np.cos((2 * np.pi * x_prime / wavelength) + psi)

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
    return (filters, len(scales), len(orientations))

def plot_gabor_filter_bank():

	scales = [
		(3, 3.0),
		(6, 5.0),
		(10, 8.0)
	]

	# Define orientations (in degrees)
	orientations = [0, 30, 60, 90, 120, 150]
	num_scales=len(scales)
	num_orientations=len(orientations)
	filters, _, _ = create_gabor_filter_bank(scales, orientations)
	fig, axs = plt.subplots(num_scales, num_orientations, figsize=(2*num_orientations, 2*num_scales))
	for i in range(num_scales):
		for j in range(num_orientations):
			idx = i * num_orientations + j
			axs[i, j].imshow(filters[idx], cmap='gray')
			axs[i, j].axis('off')
			axs[i, j].set_title(f"θ={j*180//num_orientations}°, λ={scales[i][0]}, σ={scales[i][1]}", fontsize=8)
	plt.suptitle("Gabor Filter Bank (Multi-scale, Multi-orientation)", fontsize=16)
	plt.tight_layout()
	plt.show()
	
if __name__ == "__main__":
    plot_gabor_filter_bank()