### -------------------------------------- function to run detection ---------------------------------------------------------

def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates
