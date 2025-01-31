import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import pywt
import cupy as cp
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch

SCAN_SPEED = 0.25  # Microns per second
FPS = 30


# Load video
video_path = "videos\\rawglasssmall.avi"


def analyze_video(video_path: str, method: str, scan_speed: float, fps: int):
    cap = cv2.VideoCapture(video_path)
    microns_per_frame = scan_speed * 1 / fps

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    frames = np.array(frames)  # Stack as 3D array (frames, height, width)
    # Convert height map to microns
    if method == "hilbert":
        height_map = hilbert_transform(frames) * microns_per_frame
    elif method == "cwt":
        height_map = cwt(frames) * microns_per_frame
    elif method == "cwt_gpu":
        height_map = cwt_gpu(frames) * microns_per_frame
    else:
        height_map = hilbert_transform(frames) * microns_per_frame
    return height_map

def hilbert_transform(frames):
    # Process each pixel through Hilbert transform to extract modulation envelope
    height_map = np.zeros(frames.shape[1:])

    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            intensity_profile = frames[:, x, y]
            analytic_signal = hilbert(intensity_profile)
            envelope = np.abs(analytic_signal)
            height_map[x, y] = np.argmax(envelope)  # Peak corresponds to height

    return height_map

def cwt(frames):
    height_map = np.zeros(frames.shape[1:])
    for x in range(frames.shape[0]):
        for y in range(frames.shape[1]):
            intensity_profile = frames[:, x, y]  # Extract pixel intensity over time
            peak_pos = find_peak_cwt(intensity_profile)
            height_map[x, y] = peak_pos  # Store detected peak as height value
    return height_map

def find_peak_cwt(signal, wavelet='morl', scales=np.arange(1, 50)):
    """Apply Continuous Wavelet Transform (CWT) and find peak in wavelet response."""
    coefficients, _ = pywt.cwt(signal, scales, wavelet)
    modulus = np.abs(coefficients)
    peak_idx = np.argmax(modulus.sum(axis=0))  # Sum across scales & find max
    return peak_idx

def cwt_gpu(frames, wavelet='db1', level=1, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    T, H, W = frames.shape
    video_2d = frames.reshape(T, -1).T
    tensor_data = torch.from_numpy(video_2d).to(device)  # (B, T)
    B, T_ = tensor_data.shape

    # pytorch_wavelets provides DWT1DForward for 1D signals
    dwt1d = DWT1DForward(J=level, wave=wavelet).to(device)
    # (Optional) If you plan on inverse transforms:
    # idwt1d = DWT1DInverse(wave=wavelet).to(dev)

    # The 1D forward transform in pytorch_wavelets expects input of shape: (N, C, L)
    # where N = batch size, C = number of channels, L = signal length.
    # We can treat each pixel as N= B, with a single channel (C=1).
    
    tensor_data = tensor_data.unsqueeze(1)  # => (B, 1, T)

    # Output: (yl, yh_list), 
    #   yl: the approximation at the coarsest scale, shape (B, 1, length_of_coarsest_scale)
    #   yh_list: a list of detail coefficients at each scale
    yl, yh_list = dwt1d(tensor_data)

    # Approximation at the final level:
    # shape is (B, 1, A) => (H*W, 1, A)
    
    # We want => (H, W, A)
    yl_reshaped = yl.squeeze(1).reshape(H, W, -1)  # => (H, W, A)
    
    # Detail coefficients: a list of length = 'level'
    # Each element has shape (B, 1, D_i), i.e. (H*W, 1, D_i)
    detail_coeffs = []
    for i, detail_i in enumerate(yh_list):
        # detail_i => (B, 1, D_i)
        reshaped = detail_i.squeeze(1).reshape(H, W, -1)  # => (H, W, D_i)
        detail_coeffs.append(reshaped)
    
    # ---- Step 6: Example of analyzing the wavelet results ----
    # E.g., compute the mean of the detail coefficients at each scale
    for lvl, dcoeff in enumerate(detail_coeffs, start=1):
        mean_val = dcoeff.mean().item()
        print(f"Detail scale {lvl}: mean={mean_val:.4f}")

    return height_map

def plot_height_map(height_map):
    plt.imshow(height_map, cmap='jet')
    plt.colorbar(label="Height (microns)")
    plt.title("Reconstructed Height Map")
    plt.show()

if __name__ == "__main__":
    height_map = analyze_video(video_path, method="cwt_gpu", scan_speed=SCAN_SPEED, fps=FPS)
    plot_height_map(height_map)

