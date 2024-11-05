from .club import Club

import os
from sklearn.cluster import KMeans
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club, images_to_save: int = 0, images_save_path: Optional[str] = None) -> None:
        """
        Initializes the ClubAssigner with club information and image saving parameters.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
            images_to_save (int): The number of images to save for analysis.
            images_save_path (Optional[str]): The directory path to save images.
        """
        self.club1 = club1
        self.club2 = club2
        self.club_colors: Dict[str, Any] = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color,
            'unknown': (0, 0, 0)
        }
        self.goalkeeper_colors: Dict[str, Any] = {
            club1.name: club1.goalkeeper_jersey_color,
            club2.name: club2.goalkeeper_jersey_color
        }

        # Saving images for analysis
        self.images_to_save = images_to_save
        self.output_dir = images_save_path

        if not images_save_path:
            images_to_save = 0
            self.saved_images = 0
        else:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        
            self.saved_images = len([name for name in os.listdir(self.output_dir) if name.startswith('player')])


    def save_player_image(self, img: np.ndarray, player_id: int, is_goalkeeper: bool = False) -> None:
        """
        Save the player's image to the specified directory.

        Args:
            img (np.ndarray): The image of the player.
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.
        """
        # Use 'goalkeeper' or 'player' prefix based on is_goalkeeper flag
        prefix = 'goalkeeper' if is_goalkeeper else 'player'
        filename = os.path.join(self.output_dir, f"{prefix}_{player_id}.png")
        if os.path.exists(filename):
            return
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} image: {filename}")
        # Increment the count of saved images
        self.saved_images += 1

    def get_crops(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, is_goalkeeper: bool = False) -> Tuple[int, int, int]:
        """
        Extract the jersey color from a player's bounding box in the frame.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Tuple[int, int, int]: The jersey color in BGR format.
        """
        # Save player images only if needed
        if self.saved_images < self.images_to_save:
            img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # img_top = img[0:img.shape[0] // 2, :] 
            self.save_player_image(img, player_id, is_goalkeeper)  # Pass is_goalkeeper here

        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        return img

    def get_player_club(self,team_classifier, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, is_goalkeeper: bool = False) -> Tuple[str, int]:
        """
        Determine the club associated with a player based on their jersey color.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Tuple[str, int]: The club name and the predicted class index.
        """
        crop = self.get_crops(frame, bbox, player_id, is_goalkeeper)

        pred = team_classifier.predict([crop])
        cluster_label = pred[0]
        return cluster_label, 0

    def assign_clubs(self, players_team_assigned, team_classifier,frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        Assign clubs to players and goalkeepers based on their jersey colors.

        Args:
            frame (np.ndarray): The current video frame.
            tracks (Dict[str, Dict[int, Any]]): The tracking data for players and goalkeepers.

        Returns:
            Dict[str, Dict[int, Any]]: The updated tracking data with assigned clubs.
        """
        tracks = tracks.copy()
        cluster_to_club = {0: 'Club1', 1: 'Club2'}
        for track_type in ['goalkeeper', 'player']:
            for player_id, track in tracks[track_type].items():
                if player_id in players_team_assigned:
                # Use the previously assigned club
                    club = players_team_assigned[player_id]
                else:
                    bbox = track['bbox']
                    is_goalkeeper = (track_type == 'goalkeeper')
                    cluster_label, _ = self.get_player_club(team_classifier,frame, bbox, player_id, is_goalkeeper)
                    club = cluster_to_club.get(cluster_label, 'unknown')
                    players_team_assigned[player_id] = club
                tracks[track_type][player_id]['club'] = club
                tracks[track_type][player_id]['club_color'] = self.club_colors[club]
        
        return tracks

