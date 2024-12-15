import subprocess
import os

def convert_sec_to_mp4(input_file, output_file):
    """
    Convert a .sec video file to .mp4 using FFmpeg.
    :param input_file: Path to the input .sec file
    :param output_file: Path to the output .mp4 file
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        # FFmpeg command to convert .sec to .mp4
        print("input file: ", input_file)
        command = [
            "ffmpeg",
            "-i", input_file,          # Input file
            "-c:v", "libx264",         # High-quality video codec
            "-preset", "fast",         # Balanced speed and quality
            "-crf", "18",              # High quality (lower CRF = better quality) - have done quantization instead
            # "-qp", "0",               # this results in a very large file size
            "-threads", "auto",           # Use multiple CPU threads (adjust based on your system)
            "-movflags", "faststart",  # Optimize for web streaming
            output_file                # Output file
            ]
    




        # Execute the FFmpeg command
        print(f"Converting {input_file} to {output_file}...")
        subprocess.run(command, check=True)
        print(f"Conversion complete: {output_file}")

    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e)

# Example usage
input_sec_file = r"C:\Users\hp-15\Disc D\Fire-Detection-Toyota\Fire-Detection-Toyota\sec_videos\sec1.sec"  # Replace with your .sec file
output_mp4_file = r"C:\Users\hp-15\Disc D\Fire-Detection-Toyota\Fire-Detection-Toyota\sec_videos\sec1.mp4"  # Replace with the desired output filename
convert_sec_to_mp4(input_sec_file, output_mp4_file)
