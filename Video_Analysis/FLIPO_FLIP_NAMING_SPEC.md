# Universal Naming Convention: Flipo Flip Experimental Assets

This schema applies to all files generated during the Flipo Flip simulation and experimentation, including **Videos (.mp4)**, **Data Logs (.csv)**, and **Analysis Images (.png, .jpg)**.

## 1. Naming Structure
Format: `{Color}_D{Infill}_T{Thickness}_L{EdgeWidth}_W{Weight}.{extension}`

### Parameter Definitions:
- **{Color}**: Material color (e.g., Green, Pink, Blue). Primary key for visual tracking.
- **D{Infill}**: 3D printing infill density percentage (e.g., D045 = 45% infill).
- **T{Thickness}**: Thickness of the Flipo Flip in millimeters (mm).
- **L{EdgeWidth}**: Width of the edge/prong in millimeters (mm).
- **W{Weight}**: Total mass of the physical/simulated object in grams (g).

## 2. Global Mapping Rule
Files with the same parameter string (e.g., `Pink_D045_T17.000_L05.100_W16.76`) represent the **same experimental trial**. 
- `.mp4`: The raw/processed simulation video.
- `.csv`: The physical telemetry data (time, position, velocity, etc.).
- `.png/.jpg`: Visualizations, plots, or keyframe extractions.

## 3. Extraction Regex (Universal)
Use this pattern to parse parameters regardless of file extension:
^([a-zA-Z]+)_D(\d+)_T([\d\.]+)_L([\d\.]+)_W([\d\.]+)

## 4. Examples
- `Green_D015_T16.250_L03.071_W09.87.mp4`
- `Green_D015_T16.250_L03.071_W09.87.csv`
- `Pink_D045_T17.000_L05.100_W16.76.png`