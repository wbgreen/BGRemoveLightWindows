# AI Background Remover for Windows

A lightweight, self-contained desktop application that strips the background from
images without impacting their visual quality. It uses the compact U2NetP model
running through ONNX Runtime and ships with a friendly Tkinter interface in
addition to a simple command-line utility.

## Features

- ✔️ Fast, high-quality background removal using a 4.4 MB neural network.
- ✔️ Works fully offline once installed – the ONNX model is bundled.
- ✔️ Keeps your original resolution and saves as transparent PNGs.
- ✔️ Simple graphical interface for Windows plus a scriptable CLI.

## Installation

1. Install Python 3.9 or newer for Windows.
2. Install the dependencies:

   `bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   `

   The project depends only on Pillow, NumPy, and ONNX Runtime.

## Usage

### Graphical App

Launch the desktop application with:

`bash
python -m background_remover.gui
`

1. Click **Open Image…** and choose a photo.
2. Press **Remove Background** to run the AI model.
3. Use **Save Result…** to export a transparent PNG while preserving the
   original resolution and crispness.

### Command Line

To process a file directly:

`bash
python -m background_remover.cli path/to/photo.jpg
`

The transparent output is saved alongside the original (e.g.
`photo_transparent.png`). Supply `--output` to control the destination or
`--providers` to force a specific ONNX Runtime execution provider, such as
`CUDAExecutionProvider` for GPU acceleration.

## Packaging for Windows

To ship a single-file executable for Windows, install `pyinstaller` and run:

`bash
pyinstaller --onefile --add-data "background_remover/models/u2netp.onnx;background_remover/models" -m background_remover.gui
`

This produces a standalone binary under `dist/` that includes the ONNX model.

## License

The U<sup>2</sup>-Net model weights are distributed under their original MIT
License. See the upstream project for details.
