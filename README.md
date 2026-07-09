<p align="center">
  <img src="logo.png" alt="Audio Wave Visualizer Logo" width="320"/>
</p>

# Audio Visualizer

A ray marched audio visualizer built using **Rust**, **eframe (egui + wgpu)**, **rodio**, and **rustfft**. It also has persistent state, mostly :P
Drag and drop an audio file and everything should start automatically.

## Installation & Setup

1. Make sure you have the [Rust toolchain](https://rustup.rs/) installed.
2. Clone or navigate to the repository directory.

## Running the Application

```bash
python run.py
```

Or run via Cargo:

```bash
cargo run --release
```

### You could download the binary from the releases. If you go that route:

If Windows SmartScreen warns you with *"Windows protected your PC"* when downloading or running the compiled `.exe`:
1. Click **"More info"**.
2. Click **"Run anyway"**.

This warning appears because the binary is compiled locally / open-source and is not signed with a paid developer certificate.
