from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner
from utils import rgb_bgr_converter

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    """
    A video processor for football footage that tracks objects and keypoints, assigns the ball to player, calculates the ball possession 
    and adds various annotations.
    """

    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker, 
                 club_assigner: ClubAssigner,
                 top_down_keypoints: np.ndarray, field_img_path: str, 
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:
        """
        Initializes the video processor with necessary components for tracking, annotations, and saving tracks.

        Args:
            obj_tracker (ObjectTracker): The object tracker for tracking players and balls.
            kp_tracker (KeypointsTracker): The keypoints tracker for detecting and tracking keypoints.
            club_assigner (ClubAssigner): Assigner to determine clubs for the tracked players.
            top_down_keypoints (np.ndarray): Keypoints to map objects to top-down positions.
            field_img_path (str): Path to the image of the football field used for projection.
            save_tracks_dir (Optional[str]): Directory to save tracking information. If None, no tracks will be saved.
            draw_frame_num (bool): Whether or not to draw current frame number on the output video.
        """

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator() 

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)
        
        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        
        self.frame_num = 0

        self.field_image = field_image


    def process_for_TM(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)

        # Process each frame in the batch
        all_tracks_list = []
        for idx, (frame, object_detection) in enumerate(zip(frames, batch_obj_detections)):
            
            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)

            # Assign clubs to players based on their tracked position
            # obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

            all_tracks = {'object': obj_tracks}
            all_tracks_list.append(all_tracks)



        return all_tracks_list

    def process(self,players_team_assigned, team_classifier,frames: List[np.ndarray], fps: float = 1e-6, ) -> List[np.ndarray]:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)

        processed_frames = []
        vornoi_frames = []

        # Process each frame in the batch
        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):
            
            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)

            # Assign clubs to players based on their tracked position
            
            obj_tracks = self.club_assigner.assign_clubs(players_team_assigned,team_classifier,frame, obj_tracks)

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            all_tracks = self.obj_mapper.map(all_tracks)

            

            
            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            self.frame_num += 1

            # Annotate the current frame with the tracking information
            annotated_frame, projection_frame = self.annotate(frame, all_tracks)

            # Append the annotated frame to the processed frames list
            processed_frames.append(annotated_frame)
            vornoi_frames.append(projection_frame)

        return processed_frames, vornoi_frames

    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the given frame with analised data

        Args:
            frame (np.ndarray): The current video frame to be annotated.
            tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.

        Returns:
            np.ndarray: The annotated video frame.
        """
         
        # Draw the frame number if required
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(frame, {'frame_num': self.frame_num})
        
        # Annotate the frame with keypoint and object tracking information
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # Project the object positions onto the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image, tracks['object'])

        # Combine the frame and projection into a single canvas
        combined_frame = self._combine_frame_projection(frame, projection_frame)

        # Annotate possession on the combined frame

        return combined_frame, projection_frame 
    

    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        """
        Combines the original video frame with the projection of player positions on the field image.

        Args:
            frame (np.ndarray): The original video frame.
            projection_frame (np.ndarray): The projected field image with annotations.

        Returns:
            np.ndarray: The combined frame.
        """
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 30% of its original size
        scale_proj = 0.3
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame
    


    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        """
        Saves the tracking information for objects and keypoints to the specified directory.

        Args:
            all_tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.
        """
        self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
        self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])

    

    