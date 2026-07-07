import os
import sys
import subprocess


def main():
    # Set the WGPU backend to DX12 on Windows to bypass Vulkan validation warnings
    os.environ["WGPU_BACKEND"] = "dx12"

    # Set default log level if not already defined
    if "RUST_LOG" not in os.environ:
        os.environ["RUST_LOG"] = "info"

    # Capture and forward all CLI arguments to cargo run
    # Defaults to --release mode unless --debug is explicitly passed.
    args = sys.argv[1:]
    if "--debug" in args:
        args.remove("--debug")
        cmd = ["cargo", "run"] + args
    else:
        if "--release" not in args:
            args.append("--release")
        cmd = ["cargo", "run"] + args

    print(f"Executing: {' '.join(cmd)}")

    try:
        # Launch cargo run and inherit stdout/stderr/stdin
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nProcess failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(
            "\nError: 'cargo' command not found. Please make sure Rust is installed and in your PATH."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
