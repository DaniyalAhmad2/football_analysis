from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
import cv2
import numpy as np
from typing import List
import torch
from club_assignment.team import TeamClassifier

def main():

    output_video = 'output_videos/testx.mp4'
    """
    Main function to demonstrate how to use the football analysis project.
    This script will walk you through loading models, assigning clubs, tracking objects and players, and processing the video.
    """

    # 1. Load the object detection model
    # Adjust the 'conf' value as per your requirements.
    obj_tracker = ObjectTracker(
        model_id="football-players-detection-3zvbc-7ocfe/2",    # Object Detection Model Weights Path
        conf=.5,                                            # Object Detection confidence threshold
        ball_conf=.05                                        # Ball Detection confidence threshold
    )

    # 2. Load the keypoints detection model
    # Adjust the 'conf' and 'kp_conf' values as per your requirements.
    kp_tracker = KeypointsTracker(
        model_id='football-field-detection-f07vi-apxzb/1', # Keypoints Model Weights Path
        conf=.3,                                            # Field Detection confidence threshold
        kp_conf=.7,                                         # Keypoint confidence threshold
    )
    
    # 3. Assign clubs to players based on their uniforms' colors
    # Create 'Club' objects - Needed for Player Club Assignment
    # Replace the RGB values with the actual colors of the clubs.
    club1 = Club('Club1',         # club name 
                 (232, 247, 248), # player jersey color
                 (6, 25, 21)      # goalkeeper jersey color
                 )
    club2 = Club('Club2',         # club name 
                 (172, 251, 145), # player jersey color
                 (239, 156, 132)  # goalkeeper jersey color
                 )   

    # Create a ClubAssigner Object to automatically assign players and goalkeepers 
    # to their respective clubs based on jersey colors.
    club_assigner = ClubAssigner(club1, club2)

    # 4. Initialize the BallToPlayerAssigner object
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 5. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
    # These are used to transform the perspective of the field.
    top_down_keypoints = np.array([
        [20, 16], [19, 184], [19, 341], [19, 600], [19, 757], [19, 925],             # 0-5 (left goal line)
        [95, 341], [95, 600],                                                # 6-7 (left goal box corners)
        [175, 472],                                                           # 8 (left penalty dot)
        [253, 184], [253, 341], [253, 600], [253, 757],                           # 9-12 (left penalty box)
        [733, 16], [733, 341], [733, 600], [733, 925],                        # 13-16 (halfway line)
        [1211, 184], [1211, 341], [1211, 600], [1211, 757],                       # 17-20 (right penalty box)
        [1289, 472],                                                          # 21 (right penalty dot)
        [1370, 341], [1370, 600],                                              # 22-23 (right goal box corners)
        [1446, 16], [1446, 184], [1446, 341], [1446, 600], [1446, 757], [1446, 925], # 24-29 (right goal line)
        [604, 472], [861, 472]                                               # 30-31 (center circle leftmost and rightmost points)
    ])
 
    # 6. Initialize the video processor
    # This processor will handle every task needed for analysis.
    processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                       kp_tracker,                                    # Created KeypointsTracker object
                                       club_assigner,                                 # Created ClubAssigner object
                                       ball_player_assigner,                          # Created BallToPlayerAssigner object
                                       top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                       field_img_path='input_videos/field.jpg', # Top-Down field image path
                                       save_tracks_dir='output_videos',               # Directory to save tracking information.
                                       draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                      #the output video.
                                       )
    cap = cv2.VideoCapture('573e61_0.mp4')
    frame_stop_count = 0
    frames = []
    while frame_stop_count < 500:
        ret, frame = cap.read()
        frames.append(frame)
        frame_stop_count += 1
        if not ret:
            break
        
    cap.release()
    tracks = processor.process_for_TM(frames)
    crops: List[np.ndarray] = []

    # Iterate over the frames and corresponding detections
    for frame, detection in zip(frames, tracks):
        players = detection['object']['player']
        for player_id, player_data in players.items():
            bbox = player_data['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            # Ensure the coordinates are integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Ensure coordinates are within the frame boundaries
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            # Crop the whole bounding box from the frame
            crop = frame[y1:y2, x1:x2]
            # Check if the crop is valid (non-empty)
            # cv2.imshow('crop', crop)
            # cv2.waitKey(0)
            if crop.size > 0:
                crops.append(crop)
    # print(len(crops))
    # cv2.destroyAllWindows()
    team_classifier = TeamClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        # print(tracks)
    team_classifier.fit(crops)

    

    
    # 7. Process the video
    # Specify the input video path and the output video path. 
    # The batch_size determines how many frames are processed in one go.




    # process_video(processor,                                # Created FootballVideoProcessor object
    #               video_source='573e61_0.mp4', # Video source (in this case video file path)
    #               output_video=output_video,    # Output video path (Optional)
    #               batch_size=10                           # Number of frames to process at once
    #               )


if __name__ == '__main__':
    main()
