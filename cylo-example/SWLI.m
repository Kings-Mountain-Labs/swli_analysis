clc; close all; clear
format short


tpf = 1/30; %fps
vel = 0.25; %scan speed, micron per second
muperframe = vel*tpf

v = VideoReader("betterglassmr.mp4");
i = 1;
length = v.NumFrames();
while i<=length %hasFrame(v)
    frames(:,:,i) = double(rgb2gray(readFrame(v))); %rgb2gray(
    i=i+1;
end
frameindex = [1:1:length];
tempposition = linspace(0,length*muperframe,length);
frame = frames(:,:,50);
figure(1)
s = surf(frame);
s.EdgeColor = 'none';

i = 1;
k = 1;
while k<v.Width
    i = 1;
    while i<=v.Height

        pixtimevector = [frames(i,k,:)];
        pixtimevector = reshape(pixtimevector, 1, length);
        env = envelope(pixtimevector,150);
        %plot(frameindex,pixtimevector, frameindex, env)
        %[M,I] = max(env);
        %pixdisp = tempposition(I);
        envmidline = sum(frameindex.*env) / sum(env);
        pixdisp = envmidline*muperframe;
        output(i,k) = pixdisp;

        i=i+1;
    end
    k = k+1;
end 


i = 400;
k = 400;
pixtimevector = [frames(i,k,:)];
pixtimevector = reshape(pixtimevector, 1, length);
[upenv,loenv] = envelope(pixtimevector,150);
envmidline = sum(frameindex.*upenv) / sum(upenv);
[M,I] = max(upenv);
pixdisp = tempposition(I)
disp(envmidline*muperframe)


figure(2)
tiledlayout(2,2);
nexttile;
hold on
plot(frameindex,pixtimevector, frameindex, upenv, frameindex, loenv)
plot(envmidline*[1,1], ylim(gca), 'g', 'LineWidth', 3.0);
hold off

pixtimevector = [frames(i+1,k,:)];
pixtimevector = reshape(pixtimevector, 1, length);
[upenv,loenv] = envelope(pixtimevector,150);
envmidline = sum(frameindex.*upenv) / sum(upenv);
[M,I] = max(upenv);
pixdisp = tempposition(I)
disp(envmidline*muperframe)
nexttile;
hold on
plot(frameindex,pixtimevector, frameindex, upenv, frameindex, loenv)
plot(envmidline*[1,1], ylim(gca), 'g', 'LineWidth', 3.0);
hold off

pixtimevector = [frames(i,k-1,:)];
pixtimevector = reshape(pixtimevector, 1, length);
[upenv,loenv] = envelope(pixtimevector,150);
envmidline = sum(frameindex.*upenv) / sum(upenv);
[M,I] = max(upenv);
pixdisp = tempposition(I)
disp(envmidline*muperframe)
nexttile;
hold on
plot(frameindex,pixtimevector, frameindex, upenv, frameindex, loenv)
plot(envmidline*[1,1], ylim(gca), 'g', 'LineWidth', 3.0);
hold off

pixtimevector = [frames(i+1,k-1,:)];
pixtimevector = reshape(pixtimevector, 1, length);
[upenv,loenv] = envelope(pixtimevector,150);
envmidline = sum(frameindex.*upenv) / sum(upenv);
[M,I] = max(upenv);
pixdisp = tempposition(I)
disp(envmidline*muperframe)
nexttile;
hold on
plot(frameindex,pixtimevector, frameindex, upenv, frameindex, loenv)
plot(envmidline*[1,1], ylim(gca), 'g', 'LineWidth', 3.0);
hold off


% diffenv = upenv-loenv;
% figure(3)
% plot(frameindex,diffenv)


%output = smoothdata(output);
%output = smooth(output,0.1,'rloess');
% 

figure(3)
s = surf(output);
s.EdgeColor = 'none';


output = detrend(detrend(output')');
figure(4)
s = surf(output);
s.EdgeColor = 'none';

pixtimevector = pixtimevector-mean(pixtimevector);
FD = fft(pixtimevector);
figure(5)
tiledlayout(1,3)
nexttile()
plot(pixtimevector)
title("Signal")
nexttile()
plot(abs(FD(1:140)))
title("Cropped Fourier magnitude")
nexttile()
plot(angle(FD(1:140)))
title("Cropped Fourier phase")

%output = single(output);

%OUTPUT AS TIFF
% fileName = 'MOOREBLOCK.tif';
% tiffObject = Tiff(fileName, 'w');
% % Set tags.
% tagstruct.ImageLength = size(output,1); 
% tagstruct.ImageWidth = size(output,2);
% tagstruct.Compression = Tiff.Compression.None;
% tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
% tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
% tagstruct.BitsPerSample = 32;
% tagstruct.SamplesPerPixel = size(output,3);
% tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
% tiffObject.setTag(tagstruct);
% % Write the array to disk.
% tiffObject.write(output);
% tiffObject.close;
% % Recall image.
% m2 = imread(fileName);
% % Check that it's the same as what we wrote out.
% maxDiff = max(max(m2-output)) % Should be zero.

