#  Vinyl Track Detector

This is a relatively simple computer vision project created for personal experimentation and fun.

The goal is to detect:
- **Number of tracks**
- **Location of each track**
- **Duration of each track**

...all from a **single image of a vinyl record**.

---

## How It Works

This project uses basic image processing techniques to analyse a top-down photo of a vinyl record and extract information about its tracks. The number and duration of tracks are inferred based on groove spacing and other visual cues.

> **Note:** The current method is quite sensitive to lighting conditions, so a **custom camera rig** was built to maintain a consistent angle and minimise glare or shadowing.

---

## Future Plans

-  **Album Detection**  
  Estimate the album or side based on the track count and durations using a local or online database.

-  **Needle Control**  
  Integrate with a motorised tonearm to allow for **automated skipping** to specific tracks.

---

##  Setup Tips

To get reliable results:
- Ensure the vinyl is placed flat and centered in the frame
- Use consistent, diffused lighting (avoid direct light sources)
- A top-down image works best (orthogonal view)

---

##  Tools & Technologies

- Python & OpenCV
- Custom lighting rig + overhead camera
- (Planned) Integration with tonearm servo/motor system

---

