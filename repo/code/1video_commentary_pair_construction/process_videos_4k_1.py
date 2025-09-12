#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import re
import logging
import argparse
from datetime import datetime
from pathlib import Path

def process_subject(subject, unit_id, base_dir, output_base):
    """
    Process video and step files for a specific unit within a subject.

    Args:
        subject: Subject name (e.g., "chemistry", "physics", "biology", etc.)
        unit_id: Unit ID (e.g., "ch_94", "phy_189", "bio_35", etc.)
        base_dir: Base directory for source data.
        output_base: Base directory for output data.

    Returns:
        tuple: (problem_videos, log_file, error_log)
               A dictionary mapping problem video filenames to lists of problematic step numbers,
               path to the main log file, and path to the error log file.
    """
    # Set paths
    text_dir = os.path.join(base_dir, "dataset", subject, unit_id, "step")
    video_dir = os.path.join(base_dir, "source_video", subject, unit_id)

    # Create output directory structure
    output_dir = os.path.join(output_base, "processed", subject, unit_id)
    json_base_dir = os.path.join(output_dir, "json")
    video_base_dir = os.path.join(output_dir, "video")
    os.makedirs(json_base_dir, exist_ok=True)
    os.makedirs(video_base_dir, exist_ok=True)

    # Set up logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"video_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    error_log = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Create a dedicated error logger
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.WARNING)
    error_handler = logging.FileHandler(error_log)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(error_handler)

    # Dictionary to record problematic videos and steps
    problem_videos = {}

    # Check if source directories exist
    if not os.path.exists(text_dir):
        logging.error(f"Step directory does not exist: {text_dir}")
        return problem_videos, log_file, error_log

    if not os.path.exists(video_dir):
        logging.error(f"Video directory does not exist: {video_dir}")
        return problem_videos, log_file, error_log

    # Get all JSON files
    json_files = [f for f in os.listdir(text_dir) if f.endswith('.json')]
    json_files.sort()  # Ensure consistent file order

    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mkv'))]

    logging.info(f"Starting processing for {subject}/{unit_id}, found {len(json_files)} JSON files and {len(video_files)} video files")

    # Process each JSON file and its corresponding video
    for json_file in json_files:
        # Extract the numerical part and title from the filename, supporting both hyphen and underscore formats
        # e.g., "01-三价铁离子的检验.json" -> "01", "三价铁离子的检验"
        # or "01_水的沸腾.json" -> "01", "水的沸腾"
        match = re.match(r'(\d+)([-_])(.*?)\.json', json_file)
        if not match:
            logging.warning(f"Could not parse filename: {json_file}, skipping")
            continue

        unit_number = match.group(1)
        separator = match.group(2)  # Get the separator (- or _)
        unit_name = match.group(3)

        # Find the matching video file
        matched_video = None
        for video_file in video_files:
            # Check if the video file starts with the same number and separator
            if video_file.startswith(f"{unit_number}{separator}"):
                # Further check if the title is contained in the video filename
                video_title_parts = video_file.split(separator)[1:] if len(video_file.split(separator)) > 1 else ""
                video_title = "".join(video_title_parts)
                # Basic check for title presence, ignoring dots
                if unit_name.replace(".", "") in video_title.replace(".", "") or video_title.replace(".", "") in unit_name.replace(".", ""):
                     matched_video = video_file
                     break

        # If no exact match found, try matching only by number
        if not matched_video:
            for video_file in video_files:
                 # Check if the video file starts with the same number regardless of separator
                 if re.match(rf"^{unit_number}[-_]", video_file):
                    matched_video = video_file
                    break


        if not matched_video:
            logging.warning(f"Could not find corresponding video file for: {unit_number}{separator}{unit_name}, skipping")
            continue

        video_path = os.path.join(video_dir, matched_video)

        # Create unit-specific subdirectories - using hyphen (-) for consistency in output
        unit_json_dir = os.path.join(json_base_dir, f"{unit_number}-{unit_name}")
        unit_video_dir = os.path.join(video_base_dir, f"{unit_number}-{unit_name}")
        os.makedirs(unit_json_dir, exist_ok=True)
        os.makedirs(unit_video_dir, exist_ok=True)

        # Read the JSON file
        json_path = os.path.join(text_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file {json_file}: {e}, skipping")
            error_logger.error(f"JSON decode error in {json_file}: {e}")
            continue


        # Get base name (without extension) for naming clips
        base_name = f"{unit_number}-{unit_name}"

        logging.info(f"Processing: {base_name} (Video: {matched_video})")

        # Add title information (if available in JSON)
        title_zh = data.get("title_zh", unit_name)
        title_en = data.get("title_en", "")

        # Ensure the steps field exists
        if "steps" not in data:
            logging.warning(f"Warning: 'steps' field not found in {json_file}, skipping")
            continue

        # Split the video and process each step
        has_problem = False
        for i, step in enumerate(data["steps"]):
            # Ensure required fields exist in the step
            if "startTime" not in step or "endTime" not in step:
                logging.warning(f"Warning: Step {i+1} missing time information in {json_file}, skipping")
                has_problem = True # Mark the file as having a problem
                if matched_video not in problem_videos:
                    problem_videos[matched_video] = []
                if i + 1 not in problem_videos[matched_video]:
                     problem_videos[matched_video].append(i + 1)
                continue

            start_time = step["startTime"]
            end_time = step["endTime"]
            step_zh = step.get("step_zh", f"Step {i+1}") # Default to "步骤 X" if not found
            step_en = step.get("step_en", f"Step {i+1}") # Default to "Step X" if not found

            # Build output filename
            clip_name = f"{base_name}_step_{i+1}.mp4"
            clip_path = os.path.join(unit_video_dir, clip_name)

            # Use FFmpeg to split the video
            # Note: -ss before -i for faster seeking might not be frame accurate,
            #       but -ss after -i can be slow. Using -ss after -i for accuracy.
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", start_time,
                "-to", end_time,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental", # Required for aac
                "-y",  # Overwrite existing file
                clip_path
            ]

            try:
                # Capture stdout and stderr
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                stdout, stderr = process.communicate()

                # Check if command executed successfully
                if process.returncode != 0:
                    logging.warning(f"Warning: Video {matched_video} step {i+1} extraction problem")
                    error_logger.error(f"Video {matched_video} step {i+1} failed processing (Times: {start_time}-{end_time}):\n{stderr}")
                    has_problem = True

                    # Record specific problem step
                    if matched_video not in problem_videos:
                        problem_videos[matched_video] = []
                    if i + 1 not in problem_videos[matched_video]:
                         problem_videos[matched_video].append(i + 1)


                # Check if the file was created and is not empty
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    logging.info(f"  Step {i+1} video extracted successfully: Times: {start_time} - {end_time}")

                    # Check the number of frames in the generated video
                    frame_cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-count_packets", # Use count_packets which is more reliable for streams
                        "-show_entries", "stream=nb_read_packets", # Use nb_read_packets
                        "-of", "csv=p=0",
                        clip_path
                    ]
                    try:
                         # Execute ffprobe and capture output
                        frame_count_output = subprocess.check_output(frame_cmd, stderr=subprocess.STDOUT, universal_newlines=True).strip()
                        frame_count = int(frame_count_output)
                        logging.info(f"  Extracted {frame_count} frames")

                        if frame_count == 0:
                             logging.warning(f"Warning: Step {i+1} extracted zero frames for {matched_video}")
                             has_problem = True
                             if matched_video not in problem_videos:
                                 problem_videos[matched_video] = []
                             if i + 1 not in problem_videos[matched_video]:
                                  problem_videos[matched_video].append(i + 1)

                    except (subprocess.CalledProcessError, ValueError) as e:
                        # Log ffprobe errors as warnings or errors
                        logging.warning(f"  Could not get frame count for {clip_name}: {str(e)}. ffprobe output: {frame_count_output if 'frame_count_output' in locals() else 'N/A'}")
                        # It's possible the video is fine but ffprobe failed,
                        # but zero frames definitely indicate a problem.
                        # We already added to problem_videos if ffmpeg failed.
                        pass # Don't mark as problem just for failed ffprobe unless ffmpeg failed
                else:
                    logging.warning(f"  Step {i+1} generated an invalid or empty video file for {matched_video}. FFmpeg stderr:\n{stderr}")
                    has_problem = True
                    if matched_video not in problem_videos:
                        problem_videos[matched_video] = []
                    if i + 1 not in problem_videos[matched_video]:
                         problem_videos[matched_video].append(i + 1)

            except FileNotFoundError:
                 logging.error("Error: FFmpeg or FFprobe command not found. Please ensure they are installed and in your PATH.")
                 # Exit or handle this critical error appropriately
                 return problem_videos, log_file, error_log # Critical error, stop processing unit
            except Exception as e:
                logging.error(f"  Step {i+1} processing failed with exception: {e}")
                error_logger.error(f"Video {matched_video} step {i+1} processing exception: {str(e)}")
                has_problem = True
                if matched_video not in problem_videos:
                    problem_videos[matched_video] = []
                if i + 1 not in problem_videos[matched_video]:
                     problem_videos[matched_video].append(i + 1)
                continue # Continue to next step


            # Create JSON file with metadata, including absolute paths to the clipped video
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0: # Only create metadata if video clip exists and is not empty
                metadata_file = os.path.join(unit_json_dir, f"{base_name}_step_{i+1}.json")
                metadata = {
                    "title_zh": title_zh,
                    "title_en": title_en,
                    "original_video": os.path.abspath(video_path),  # Use absolute path
                    "clip_start": start_time,
                    "clip_end": end_time,
                    "step_zh": step_zh,
                    "step_en": step_en,
                    "step_number": i+1,
                    "clip_video_path": os.path.abspath(clip_path),  # Use absolute path
                    "subject": subject  # Add subject information
                }
                try:
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    logging.info(f"  Generated metadata file: {metadata_file}")
                except IOError as e:
                     logging.error(f"Error writing metadata file {metadata_file}: {e}")
                     error_logger.error(f"Error writing metadata for {matched_video} step {i+1}: {e}")
                     has_problem = True
                     if matched_video not in problem_videos:
                         problem_videos[matched_video] = []
                     if i + 1 not in problem_videos[matched_video]:
                         problem_videos[matched_video].append(i + 1)


        # If there were any problems in this video's steps, log a summary warning
        if has_problem:
            logging.warning(f"Video {matched_video} has problematic steps: {problem_videos.get(matched_video, [])}")


    # Write problematic videos to a JSON file for later analysis
    if problem_videos:
        problem_file = os.path.join(output_dir, f"problem_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(problem_file, 'w', encoding='utf-8') as f:
                json.dump(problem_videos, f, ensure_ascii=False, indent=2)
            logging.info(f"Problematic video information saved to: {problem_file}")
        except IOError as e:
            logging.error(f"Error writing problem video summary file {problem_file}: {e}")
            error_logger.error(f"Error writing problem video summary: {e}")


    logging.info(f"Processing complete! Found {len(json_files)} JSON files, Number of videos with problems: {len(problem_videos)}")

    return problem_videos, log_file, error_log

def main():
    parser = argparse.ArgumentParser(description='Process 4k video and step files')
    parser.add_argument('--subject', type=str, required=True, help='Subject name (e.g., chemistry, physics, biology, etc.)')
    parser.add_argument('--unit_id', type=str, required=True, help='Unit ID (e.g., ch_94, phy_189, bio_35, etc.)')
    # Changed default paths to generic placeholders
    parser.add_argument('--base_dir', type=str, default='/path/to/source_data', help='Base directory for source data (e.g., /home/user/Dataset/Exp-Com/4k/4k_0)')
    parser.add_argument('--output_base', type=str, default='/path/to/output_data', help='Base directory for output data (e.g., /home/user/Dataset/Exp-Com/4k/4k_1)')

    args = parser.parse_args()

    print(f"Starting processing for {args.subject}/{args.unit_id}")
    problem_videos, log_file, error_log = process_subject(
        args.subject,
        args.unit_id,
        args.base_dir,
        args.output_base
    )

    print(f"Processing complete! Detailed log saved to {log_file}")
    print(f"Error log saved to {error_log}")
    print(f"Number of videos with problems: {len(problem_videos)}")
    print(f"Total number of problematic steps across all videos: {sum(len(steps) for steps in problem_videos.values())}")

if __name__ == "__main__":
    main()