import cv2
import os
import glob
import queue
import threading
import tempfile
import time
import signal
import traceback
from typing import List, Tuple, Optional
import numpy as np


def _convert_frames_to_video(frame_dir: str, output_video: str, fps: float, frame_size: Tuple[int, int]) -> None:
    """
    Convert frames in a directory to video files.

    Args:
        frame_dir (str): Directory containing frame images.
        output_video (str): Path to save the football analysis video.
        fps (float): Frames per second for the output videos.
        frame_size (Tuple[int, int]): Size of the football analysis frames as (width, height).
    """
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    # Initialize VideoWriter for football analysis frames
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    # Get sorted lists of frame filenames
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    vornoi_files = sorted(glob.glob(os.path.join(frame_dir, "frame_vornoi_*.jpg")))

    if not frame_files:
        out.release()
        print("There are no football analysis frames to save.")
        return

    # Read the first Voronoi frame to get its size
    if vornoi_files:
        first_voronoi_frame = cv2.imread(vornoi_files[0])
        if first_voronoi_frame is not None:
            voronoi_frame_size = (first_voronoi_frame.shape[1], first_voronoi_frame.shape[0])
            # Initialize VideoWriter for Voronoi frames with the correct size
            out2 = cv2.VideoWriter("output_videos/Voronoi_out.mp4", fourcc, fps, voronoi_frame_size)
        else:
            print("Could not read the first Voronoi frame to determine frame size.")
            out.release()
            return
    else:
        print("No Voronoi frames found.")
        out.release()
        return

    for filename, vornoiname in zip(frame_files, vornoi_files):
        img = cv2.imread(filename)
        img2 = cv2.imread(vornoiname)

        if img is not None:
            out.write(img)
        else:
            print(f"Could not read football analysis frame {filename}")

        if img2 is not None:
            out2.write(img2)
        else:
            print(f"Could not read Voronoi frame {vornoiname}")

    out.release()
    out2.release()
    print(f"Videos saved as {output_video} and output_videos/Voronoi_out.mp4")


def process_video(players_team_assigned,team_classifier,processor = None, video_source: str = 0, output_video: Optional[str] = "output.mp4", 
                  batch_size: int = 30, skip_seconds: int = 0) -> None:
    """
    Process a video file or stream, capturing, processing, and displaying frames.

    Args:
        processor (AbstractVideoProcessor): Object responsible for processing frames.
        video_source (str, optional): Video source (default is "0" for webcam).
        output_video (Optional[str], optional): Path to save the output video or None to skip saving.
        batch_size (int, optional): Number of frames to process at once.
        skip_seconds (int, optional): Seconds to skip at the beginning of the video.
    """
    from annotation import AbstractVideoProcessor  # Lazy import

    if processor is not None and not isinstance(processor, AbstractVideoProcessor):
        raise ValueError("The processor must be an instance of AbstractVideoProcessor.")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = int(skip_seconds * fps)

    # Skip the first 'frames_to_skip' frames
    for _ in range(frames_to_skip):
        cap.read()  # Simply read and discard the frames

    frame_queue = queue.Queue(maxsize=100)
    processed_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Signal handler to initiate shutdown on interrupt."""

        print("Interrupt received, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    
    def frame_capture_thread() -> None:
        """Thread to capture frames from the video source."""

        print("Starting frame capture")
        frame_count = frames_to_skip  # Start counting frames from here
        try:
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("No more frames to capture or end of video")
                    break
                resized_frame = cv2.resize(frame, (1920, 1080))

                frame_queue.put((frame_count, resized_frame))
                frame_count += 1
        except Exception as e:
            print(f"Error in frame capture: {e}")
        finally:
            cap.release()
            frame_queue.put(None)  # Signal end of capture
        print("Frame capture complete")

    def frame_processing_thread() -> None:
        """Thread to process frames from the frame queue."""
        
        print("Starting frame processing")
        frame_batch = []
        
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=1)
                if item is None:
                    print("No more frames to process")
                    if frame_batch:
                        process_batch(frame_batch)
                    break
                frame_count, frame = item
                frame_batch.append((frame_count, frame))

                if len(frame_batch) == batch_size:
                    process_batch(frame_batch,players_team_assigned)
                    frame_batch = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")

        processed_queue.put(None)  # Signal end of processing
        print("Frame processing complete")

    def process_batch(batch: List[Tuple[int, np.ndarray]], players_team_assigned) -> None:
        """
        Process a batch of frames and put results in the processed queue.

        Args:
            batch (List[Tuple[int, np.ndarray]]): List of tuples containing frame count and frame data.
        """
        frames = [frame for _, frame in batch]
        try:
            processed_batch, vornoi_batch = processor.process(players_team_assigned,team_classifier,frames, fps)
            for (frame_count, _), processed_frame, vornoi_frame in zip(batch, processed_batch, vornoi_batch):
                processed_queue.put((frame_count, processed_frame,vornoi_frame))
        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()

    def frame_display_thread(temp_dir: str) -> None:
        """Thread to display processed frames."""

        print("Starting frame display")
        while not stop_event.is_set():
            try:
                item = processed_queue.get(timeout=1)
                if item is None:
                    print("No more frames to display")
                    break
                frame_count, processed_frame, vornoi_frame = item

                frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                vornoi_filename = os.path.join(temp_dir,f"frame_vornoi_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, processed_frame)
                cv2.imwrite(vornoi_filename, vornoi_frame)
                
                cv2.imshow('Football Analysis', processed_frame)
                cv2.imshow("Vornoi_visualization", vornoi_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, initiating shutdown")
                    stop_event.set()
                    break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error displaying frame: {e}")

        cv2.destroyAllWindows()
        print("Frame display complete")

    width = 1920
    height = 1080

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            threads = [
                threading.Thread(target=frame_capture_thread, name="Capture"),
                threading.Thread(target=frame_processing_thread, name="Processing"),
                threading.Thread(target=frame_display_thread, args=(temp_dir,), name="Display")
            ]

            for thread in threads:
                thread.start()

            # Wait for user to press 'q'
            while any(thread.is_alive() for thread in threads):
                if stop_event.is_set():
                    print("Stopping threads...")
                    break
                time.sleep(0.1)

            stop_event.set()  # Ensure all threads know to stop

            for thread in threads:
                thread.join(timeout=10)  # Give each thread 10 seconds to join
                if thread.is_alive():
                    print(f"Thread {thread.name} did not terminate gracefully")

            # Ensure all queues are empty
            while not frame_queue.empty():
                frame_queue.get()
            while not processed_queue.empty():
                processed_queue.get()

            print("All threads have completed.")
            # Only convert to video if output_video is not None
            if output_video is not None:
                print("Converting frames to video...")
                _convert_frames_to_video(temp_dir, output_video, fps, (width, height))

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Video processing completed. Program will now exit.")
    os._exit(0)  # Force exit the program
