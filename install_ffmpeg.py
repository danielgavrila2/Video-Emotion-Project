import subprocess
import sys

def install_ffmpeg():
    print("Starting FFMPEG installation")

    # 1. Install (Upgrade) pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    #2. Install setuptools
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

    #3. Install ffmpeg
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed FFMPEG successfully!")
    
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")


    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True    
        )

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFMPEG binary successfully!")

    except Exception as e:
        print(f"Failed to install static FFMPEG: {e}")

    
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("FFMPEG version:")
        print(result.stdout)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFMPEG installation verification failed!")
        return False
