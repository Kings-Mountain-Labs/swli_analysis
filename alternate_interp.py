from fft_interp import *

if __name__ == "__main__":
    # time all the calls
    start_time = time.time()
    # frames = get_frames("videos/carbide_90fps_750nmps.avi", False)
    frames = get_frames("videos/DTbrass_100fps_500nmps.avi", False)
    print(f"Time to get frames: {time.time() - start_time}")
    
    start_time = time.time()
    red, green, blue = split_rgb(frames)
    print(f"Time to split rgb: {time.time() - start_time}")
    
    start_time = time.time()
    # red_height_map = analyze_video(red, method="hilbert_gpu", scan_speed=0.75, fps=90)
    red_height_map = analyze_video(red, method="hilbert", scan_speed=0.5, fps=100)
    print(f"Time to analyze red: {time.time() - start_time}")

    # start_time = time.time()
    # blue_height_map = analyze_video(blue, method="hilbert_gpu", scan_speed=SCAN_SPEED, fps=FPS)
    # print(f"Time to analyze blue: {time.time() - start_time}")

    # start_time = time.time()
    # green_height_map = analyze_video(green, method="hilbert_gpu", scan_speed=SCAN_SPEED, fps=FPS)
    # print(f"Time to analyze green: {time.time() - start_time}")


    red_height_map = remove_maxes(remove_zeros(red_height_map))
    plot_height_map(red_height_map, extra_title="Red")
    plot_heat_map(red_height_map, extra_title="Red")

    # plot_height_map(blue_height_map, extra_title="Blue")
    # plot_height_map(green_height_map, extra_title="Green")