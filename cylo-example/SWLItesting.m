clc; close all; clear
format short



%% Dashboard
rng(4);                 % "Chosen by fair dice roll. Guaranteed to be random."
sig_length = 5000;      % Simulated signal length in samples
carrier_period = 100;    % Simulated sigmal carrier frequency period in samples
noise_amplitude = 0;    % Additive noise for simulated data
sig_width = 400;        % Approx. width of the simulated data's pulse in samples


%% Generate simulated waveform data
% I'll do this via amplitude modulation of a sinusoid, plus some additive noise
samples = (0:(sig_length-1))';
bias = @(x) exp(-(x-sig_length/3).^2/(2*sig_width^2));  % Gaussian lobe
A = .1*cumsum(size(samples)) .* bias(samples);
signal = A .* cos(2*pi*samples/carrier_period + pi/2);

%% FFT 

%degroot FDA
FD = fft(signal);
phi = angle(FD);



%% plot
figure(1)
tiledlayout(1,3)
nexttile()
plot(signal)
title("Signal")
nexttile()
plot(abs(FD(1:140)))
title("Cropped Fourier magnitude")
nexttile()
plot(angle(FD(1:140)))
title("Cropped Fourier phase")

