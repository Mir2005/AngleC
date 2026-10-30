# AngleC
**A Semi-Automated Contact-Angle Measurement System**

AngleC is a Python-based application for capturing live droplet images through an external webcam and measuring contact angles using geometric circle and ellipse fitting.

### Features
- Live webcam preview and image capture.
- User-defined image naming and automatic watermarking.
- Manual baseline and droplet-surface point selection.
- Circle- and ellipse-based contact-angle estimation.
- Visualisation of fitted curves, tangents, and angles.
- Export of annotated PNG images and TXT measurement data.

### Workflow
1. Connect the webcam and launch AngleC.
2. Press `C` to capture and save an image.
3. Select two baseline points and at least five droplet-surface points.
4. Review the fitted curves and contact-angle estimates.
5. Press `E` to export the annotated image and measurement data.

### Requirements
- Python 3.12
- NumPy 1.26.4
- OpenCV 4.9.0
- Additional: SciPy, Pillow, and Matplotlib

See [SETUP.md](SETUP.md) for installation instructions.

### Limitations
Point selection is manual; automatic droplet detection is not implemented. Lengths are reported in pixels without calibration. AngleC uses geometric fitting, not Young–Laplace fitting or automatic ADSA.

## Development Credit

**Concept and system design:** Mir Md Minhaajuddin  
**Coding assistance:** ChatGPT and Claude

AngleC uses local image processing and does not require a paid API key or cloud service for its measurement workflow.
