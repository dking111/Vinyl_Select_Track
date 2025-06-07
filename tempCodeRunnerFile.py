    vert_line = np.array(vert_line, dtype=np.float32)
    gradient = np.abs(np.diff(vert_line))
    peaks, _ = find_peaks(gradient, prominence=5)

    # Filter: keep only peaks where the intensity is <= 100
    refined_peaks = [p for p in peaks if vert_line[p + 1] <= 100]