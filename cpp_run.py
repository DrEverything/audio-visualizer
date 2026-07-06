import os
import subprocess
import sys
import shutil


def main():
    # 1. Define paths
    build_dir = "build"
    occt_bin_path = r"C:\libs\occt\opencascade-8.0.0-vc14-64\win64\vc14\bin"

    # Find all directories under the 3rdparty directory that contain .dll files
    thirdparty_root = r"C:\libs\occt\3rdparty-vc14-64"
    dll_paths = set()
    if os.path.exists(thirdparty_root):
        for root, dirs, files in os.walk(thirdparty_root):
            for file in files:
                if file.lower().endswith(".dll"):
                    dll_paths.add(root)
                    break

    # Add OpenCASCADE DLLs and all 3rdparty DLLs to the system PATH environment variable for this run
    paths_to_add = [occt_bin_path] + list(dll_paths)
    os.environ["PATH"] = (
        os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get("PATH", "")
    )

    # 2. Create build directory if it doesn't exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # 3. Configure CMake
    print("--- Configuring CMake ---")
    subprocess.run(["cmake", "-S", ".", "-B", build_dir], check=True)

    # Copy compile_commands.json to the workspace root for clangd LSP
    compile_commands_src = os.path.join(build_dir, "compile_commands.json")
    compile_commands_dst = "compile_commands.json"
    if os.path.exists(compile_commands_src):
        shutil.copy(compile_commands_src, compile_commands_dst)
        print("Copied compile_commands.json to root for clangd LSP.")

    # 4. Build the executable
    print("\n--- Building executable ---")
    subprocess.run(["cmake", "--build", build_dir, "--config", "Release"], check=True)

    # Clean up old step and stl files if they exist in root
    for old_file in ["mechanical_part.step", "mechanical_part.stl"]:
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                print(f"Removed old file: {old_file}")
            except Exception as e:
                print(f"Error removing {old_file}: {e}")

    # 5. Run the executable
    print("\n--- Running program ---")
    executable = os.path.join(build_dir, "Release", "occt_example.exe")
    if not os.path.exists(executable):
        # On some systems/generators it might be directly in the build root
        executable = os.path.join(build_dir, "occt_example.exe")

    subprocess.run([executable], check=True)

    # 6. Check if Blender is running, if not launch it with the auto-reload script
    blender_path = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
    is_blender_running = False
    try:
        tasklist_output = subprocess.check_output('tasklist', shell=True).decode('utf-8', errors='ignore')
        if "blender.exe" in tasklist_output.lower():
            is_blender_running = True
    except Exception:
        pass

    if not is_blender_running:
        if os.path.exists(blender_path):
            print("\n--- Launching Blender with Auto-Reload ---")
            # Launch Blender asynchronously so it doesn't block the terminal
            subprocess.Popen([blender_path, "--python", "blender_auto_reload.py"])
        else:
            print(
                f"\nWarning: Blender executable not found at '{blender_path}'. Please launch Blender manually with: --python blender_auto_reload.py"
            )
    else:
        print(
            "\nBlender is already running. The running instance will automatically reload the model."
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with error code: {e.returncode}")
        sys.exit(e.returncode)
