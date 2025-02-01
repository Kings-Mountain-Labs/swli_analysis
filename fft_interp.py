import numpy as np
import cv2
import plotly.express as px
from scipy.signal import hilbert, butter, filtfilt
import pywt
import cupy as cp 
# from pytorch_wavelets import DWT1DForward, DWT1DInverse
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
    average_pixel_intensity = np.mean(frames, axis=0)
    frames = frames - average_pixel_intensity
    # normalize the frames
    # frames = frames / np.max(frames) # this takes hella long
    # Convert height map to microns
    if method == "hilbert":
        height_map = hilbert_transform(frames) * microns_per_frame
    elif method == "hilbert_gpu":
        height_map = hilbert_gpu(frames) * microns_per_frame
    elif method == "cwt":
        height_map = cwt(frames) * microns_per_frame
    # elif method == "cwt_gpu":
    #     height_map = cwt_gpu(frames) * microns_per_frame
    else:
        height_map = hilbert_transform(frames) * microns_per_frame
    return height_map

def hilbert_transform(frames):
    # Process each pixel through Hilbert transform to extract modulation envelope
    # you could use numpy straight without the for loops but you need many gigs of ram (like 90 or something) and it would be slower
    height_map = np.zeros(frames.shape[1:])
    b, a = butter(2, 0.01, btype='lowpass')

    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            intensity_profile = frames[:, x, y]
            analytic_signal = hilbert(intensity_profile)
            envelope = np.abs(analytic_signal)
            filtered_envelope = filtfilt(b, a, envelope)
            # height_map[x, y] = np.argmax(envelope)  # Peak corresponds to height
            height_map[x, y] = np.argmax(filtered_envelope)
            if x == 0 and y == 0 and False:
                fig = px.line(y=envelope)
                fig.add_scatter(y=intensity_profile)
                fig.add_scatter(y=filtered_envelope)
                fig.add_scatter(x=[height_map[x, y]], y=[0])
                fig.show()
                exit()

    return height_map

def hilbert_gpu(frames) -> np.ndarray:
    hilbert_ed = hilbert_transform_1d_torch(frames, axis=0)
    hilbert_ed = hilbert_ed.cpu().numpy()
    # plot for one pixel to debug
    fig = px.line(y=hilbert_ed[:, 0, 0])
    fig.add_scatter(y=frames[:, 0, 0])
    fig.show()
    return np.argmax(hilbert_ed, axis=0)


def hilbert_transform_1d_torch(data_np: np.ndarray, axis: int = -1) -> torch.Tensor:
    """
    Compute the 1D Hilbert transform of a 3D real array along the specified axis
    using PyTorch's FFT operations.

    Parameters
    ----------
    data_np : np.ndarray
        A 3D real-valued NumPy array (e.g. shape (X, Y, Z)).
    axis : int
        The axis along which to compute the 1D Hilbert transform.

    Returns
    -------
    hilbert : np.ndarray
        A PyTorch tensor containing the Hilbert transform of data_np
        along the specified axis. The shape matches data_np, but the dtype
        is float (matching the imaginary result of the inverse FFT).
    """

    # Convert the NumPy array to a PyTorch tensor (float or double)
    # We'll assume float32 here; adjust as needed
    data_torch = torch.from_numpy(data_np).to(torch.float32)

    # FFT along the chosen axis
    data_fft = torch.fft.fft(data_torch, dim=axis)

    # Prepare the frequency-domain multiplier for the Hilbert transform
    n = data_torch.size(axis)

    # Create an empty complex filter (shape = n), initially zeros
    hilb_filter = torch.zeros(n, dtype=torch.complex64, device=data_fft.device)

    # Handle even/odd length along 'axis'
    #   - DC component (k=0) and (if even length) Nyquist freq (k=n/2) remain 0
    #   - For 1 <= k < n/2: multiply by -j
    #   - For n/2 < k < n: multiply by +j
    if n % 2 == 0:
        # Even number of points
        #  - Positive freqs are indices [1 ... n/2 - 1]
        #  - Nyquist freq is index n/2
        hilb_filter[1 : (n // 2)] = -1j
        hilb_filter[(n // 2 + 1) : ] = 1j
    else:
        # Odd number of points
        #  - Positive freqs are indices [1 ... (n-1)//2]
        #  - Negative freqs are indices [(n+1)//2 ... n-1]
        half_n = (n + 1) // 2
        hilb_filter[1 : half_n] = -1j
        hilb_filter[half_n : ] = 1j

    # Reshape the filter so it can broadcast along 'axis' in a 3D tensor
    # Build a shape of [1,1,1] and replace the dimension at 'axis' with n
    shape = [1, 1, 1]
    shape[axis] = n
    hilb_filter = hilb_filter.reshape(shape)

    # Apply the Hilbert filter in the frequency domain
    data_fft_filtered = data_fft * hilb_filter

    # Inverse FFT to get the Hilbert transform in time/space domain
    # The result is, in general, a complex tensor whose imaginary part
    # corresponds to the Hilbert transform of the original data.
    data_ifft = torch.fft.ifft(data_fft_filtered, dim=axis)

    # The Hilbert transform is the imaginary part
    hilbert_torch = data_ifft.imag

    return hilbert_torch

def cwt(frames):
    height_map = np.zeros(frames.shape[1:])
    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            print(f"Processing pixel {x}, {y} of {frames.shape[1:]}")
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



def plot_height_map(height_map, ldr=True):
    if ldr:
        # calculate the 2 sigma range of the height map and limit the color bar to that range
        two_sigma = np.std(height_map) * 2
        average_height = np.mean(height_map)
        vmin = average_height - two_sigma
        vmax = average_height + two_sigma
        print(f"vmin: {vmin}, vmax: {vmax} for height map")

    fig = px.imshow(height_map, color_continuous_scale="viridis", zmin=vmin, zmax=vmax, labels={"colorbar": "Height (microns)"})
    fig.update_layout(title="Reconstructed Height Map")
    fig.show()

if __name__ == "__main__":
    height_map = analyze_video(video_path, method="hilbert", scan_speed=SCAN_SPEED, fps=FPS)
    plot_height_map(height_map)

