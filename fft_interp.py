import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, find_peaks_cwt, savgol_filter
from scipy.ndimage import gaussian_filter1d
import torch
import time

SCAN_SPEED = 0.25  # Microns per second
FPS = 30


# Load video
video_path = "videos/rawglasssmall.avi"


def get_device():
    # check if cuda or mpl is available
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        print("cuda is available")
        device = torch.device("cuda")
    elif torch.mps.is_available() and torch.backends.mps.is_available():
        print("cuda is not available")
        device = torch.device("mps")
    else:
        print("cuda and mps are not available")
        device = "cpu"
    return device

def get_frames(filepath: str, greyscale=True) -> np.ndarray:
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if greyscale:
        frames = np.empty((total_frames, height, width), dtype=np.uint8)
    else:
        frames = np.empty((total_frames, height, width, 3), dtype=np.uint8)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if greyscale:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frames[i] = frame
    
    cap.release()
    average_pixel_intensity = np.mean(frames, axis=0)
    frames = frames - average_pixel_intensity
    # normalize the frames
    frames = frames / np.max(np.abs(frames), axis=0) # this takes hella long
    return frames

def analyze_video(frames: np.ndarray, method: str, scan_speed: float, fps: int):
    microns_per_frame = scan_speed * 1 / fps
    
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

def refine_peak(envelope, initial_peak, window=101):
    x_vals = np.arange(initial_peak - window, initial_peak + window + 1)
    y_vals = envelope[initial_peak - window:initial_peak + window + 1]
    if len(x_vals) != len(y_vals):
        # print(f"x_vals and y_vals must have the same length but are {len(x_vals)} and {len(y_vals)}")
        return initial_peak
    # Fit a quadratic function: y = ax^2 + bx + c
    coeffs = np.polyfit(x_vals, y_vals, 2)
    a, b, c = coeffs

    # The vertex (peak) of the parabola is at x = -b/(2a)
    refined_peak = -b / (2*a)
    return max(min(refined_peak, envelope.shape[0] - 1), 0)

def hilbert_transform(frames):
    # Process each pixel through Hilbert transform to extract modulation envelope
    # you could use numpy straight without the for loops but you need many gigs of ram (like 90 or something) and it would be slower
    height_map = np.zeros(frames.shape[1:])
    sos = butter(2, 0.01, btype='lowpass', output='sos')

    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            intensity_profile = frames[:, x, y]
            analytic_signal = hilbert(intensity_profile)
            envelope = np.abs(analytic_signal)
            filtered_envelope = sosfiltfilt(sos, envelope)
            # height_map[x, y] = np.argmax(envelope)  # Peak corresponds to height
            # height_map[x, y] = np.argmax(filtered_envelope)
            initial_peak = np.argmax(filtered_envelope)
            height_map[x, y] = refine_peak(envelope, initial_peak)
            if x == 0 and y == 0 and True:
                fig = px.line(y=envelope)
                fig.add_scatter(y=intensity_profile)
                fig.add_scatter(y=filtered_envelope)
                fig.add_scatter(x=[height_map[x, y]], y=[0])
                fig.show()

    return height_map

def hilbert_gpu(frames) -> np.ndarray:
    device = get_device()
    print(f"Using device: {device}")
    gpu_frames = torch.tensor(frames, dtype=torch.float32)
    gpu_frames = gpu_frames.to(device)
    # this hella needs to be batched instead of just sending the tensor
    hilbert_ed = batch_hilbert_transform(gpu_frames)
    hilbert_ed = hilbert_ed.abs()
    hilbert_ed = hilbert_ed.cpu().numpy()
    # plot for one pixel to debug


    sos = butter(2, 0.01, btype='lowpass', output='sos')

    # this next bit should be done on a GPU
    filtered_envelope = sosfiltfilt(sos, hilbert_ed, axis=0)
    # filtered_envelope = savgol_filter(hilbert_ed, window_length=151, polyorder=3, axis=0)
    height_map = np.argmax(filtered_envelope, axis=0)
    x, y = 400, 400
    # gaussian = gaussian_filter1d(hilbert_ed[:, x, y], sigma=5)
    sg = savgol_filter(hilbert_ed[:, x, y], window_length=201, polyorder=2)
    fig = px.line(y=hilbert_ed[:, x, y], title="Hilbert Transform")
    fig.add_scatter(y=frames[:, x, y], name="Original")
    fig.add_scatter(y=filtered_envelope[:, x, y], name="Filtered")
    # fig.add_scatter(y=gaussian, name="Gaussian Filter")
    fig.add_scatter(y=sg, name="Savitzky-Golay Filter")
    fig.add_scatter(x=[height_map[x, y]], y=[0], name="Peak")
    fig.show()
    return height_map


def batch_hilbert_transform(frames: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    """
    Apply Hilbert transform to a video tensor in batches.
    
    Args:
        frames: Tensor of shape (time, width, height)
        batch_size: Number of pixels to process at once
    
    Returns:
        Transformed tensor of same shape
    """
    time, width, height = frames.shape
    hilbert_transform = HilbertTransform1D().to(frames.device)
    
    # Reshape to (width*height, time)
    frames_reshaped = frames.permute(1, 2, 0).reshape(-1, time)
    
    # Process in batches
    result = []
    for i in range(0, frames_reshaped.shape[0], batch_size):
        batch = frames_reshaped[i:i+batch_size]
        transformed = hilbert_transform(batch, axis=1)
        result.append(transformed)
    
    # Combine batches and reshape back to original dimensions
    transformed_frames = torch.cat(result, dim=0)
    transformed_frames = transformed_frames.reshape(width, height, time)
    transformed_frames = transformed_frames.permute(2, 0, 1)
    
    return transformed_frames

class HilbertTransform1D(torch.nn.Module):
    def __init__(self):
        """
        Initialize the HilbertTransform module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """
        Compute the analytic signal (i.e. original signal + i * Hilbert transform)
        along the specified axis.

        Args:
            x (torch.Tensor): Input tensor (typically real-valued).
            axis (int): The axis along which to compute the Hilbert transform.
                        Defaults to the last dimension (-1).

        Returns:
            torch.Tensor: The analytic signal. The real part is the original signal,
                          and the imaginary part is its Hilbert transform.
        """
        # Compute the FFT along the specified axis.
        X = torch.fft.fft(x, dim=axis)
        n = x.shape[axis]

        # Create the multiplier for the Hilbert transform in the frequency domain.
        # This multiplier zeros out the negative frequencies and doubles the positive ones.
        h = torch.zeros(n, dtype=X.dtype, device=X.device)
        if n % 2 == 0:
            # For even-length signals:
            #   h[0] (DC) remains 1,
            #   h[1:n//2] are multiplied by 2,
            #   h[n//2] (Nyquist frequency) is kept as 1,
            #   and the rest are 0.
            h[0] = 1
            h[1:n//2] = 2
            h[n//2] = 1
        else:
            # For odd-length signals:
            #   h[0] remains 1,
            #   h[1:(n+1)//2] are multiplied by 2,
            #   and the rest are 0.
            h[0] = 1
            h[1:(n+1)//2] = 2

        # Reshape h so that it broadcasts correctly when multiplied with X.
        # Create a shape of ones and set the dimension along which FFT was taken.
        shape = [1] * X.ndim
        shape[axis] = n
        h = h.view(*shape)

        # Apply the multiplier in the frequency domain.
        X_filtered = X * h

        # Compute the inverse FFT to obtain the analytic signal.
        x_analytic = torch.fft.ifft(X_filtered, dim=axis)
        return x_analytic

def cwt(frames):
    height_map = np.zeros(frames.shape[1:])
    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            print(f"Processing pixel {x}, {y} of {frames.shape[1:]}")
            intensity_profile = frames[:, x, y]  # Extract pixel intensity over time
            analytic_signal = hilbert(intensity_profile)
            envelope = np.abs(analytic_signal)
            peak_pos = find_peaks_cwt(envelope, 600)
            height_map[x, y] = peak_pos[0]  # Store detected peak as height value
            if x == 0 and y == 0 and True:
                print(f"peak_pos: {peak_pos}")
                fig = px.line(y=intensity_profile)
                fig.add_scatter(y=envelope)
                fig.add_scatter(x=peak_pos, y=np.zeros(len(peak_pos)))
                fig.show()
                exit()
    return height_map


def plot_heat_map(height_map, extra_title="", ldr=True):
    if ldr:
        # calculate the 2 sigma range of the height map and limit the color bar to that range
        two_sigma = np.std(height_map) * 2
        average_height = np.mean(height_map)
        vmin = average_height - two_sigma
        vmax = average_height + two_sigma
        print(f"vmin: {vmin}, vmax: {vmax} for height map")

    fig = px.imshow(height_map, color_continuous_scale="viridis", zmin=vmin, zmax=vmax, labels={"colorbar": "Height (microns)"})
    fig.update_layout(title=f"Reconstructed Height Map {extra_title}")
    fig.show()

def plot_height_map(height_map, extra_title=""):
    # calculate the xy points for the heat map
    microns_per_pixel = 150/600 #this is a guess
    x, y = np.meshgrid(np.linspace(0, height_map.shape[0]*microns_per_pixel, height_map.shape[0]), np.linspace(0, height_map.shape[1]*microns_per_pixel, height_map.shape[1]))
    fig = go.Figure(data=[go.Surface(z=height_map, x=x, y=y)])
    fig.update_layout(title=f"Reconstructed Height Map {extra_title}")
    fig.show()


def split_rgb(frames):
    red = frames[:, :, :, 2]
    green = frames[:, :, :, 1]
    blue = frames[:, :, :, 0]
    return red, green, blue

if __name__ == "__main__":
    # time all the calls
    start_time = time.time()
    frames = get_frames(video_path, False)
    print(f"Time to get frames: {time.time() - start_time}")
    
    start_time = time.time()
    red, green, blue = split_rgb(frames)
    print(f"Time to split rgb: {time.time() - start_time}")
    
    start_time = time.time()
    red_height_map = analyze_video(red, method="hilbert", scan_speed=SCAN_SPEED, fps=FPS)
    print(f"Time to analyze red: {time.time() - start_time}")

    # start_time = time.time()
    # blue_height_map = analyze_video(blue, method="hilbert_gpu", scan_speed=SCAN_SPEED, fps=FPS)
    # print(f"Time to analyze blue: {time.time() - start_time}")

    # start_time = time.time()
    # green_height_map = analyze_video(green, method="hilbert_gpu", scan_speed=SCAN_SPEED, fps=FPS)
    # print(f"Time to analyze green: {time.time() - start_time}")

    
    plot_height_map(red_height_map, extra_title="Red")
    plot_heat_map(red_height_map, extra_title="Red")
    # plot_height_map(blue_height_map, extra_title="Blue")
    # plot_height_map(green_height_map, extra_title="Green")


