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
        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors: Dict[str, Any] = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color,
            'unknown': (0, 0, 0)
        }
        self.goalkeeper_colors: Dict[str, Any] = {
            club1.name: club1.goalkeeper_jersey_color,
            club2.name: club2.goalkeeper_jersey_color
        }
        self.all_club_colors = {
            club1.name: [club1.player_jersey_color, club1.goalkeeper_jersey_color],
            club2.name: [club2.player_jersey_color, club2.goalkeeper_jersey_color],
            
        }
        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)

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

    def apply_mask(self, image: np.ndarray, green_threshold: float = 0.08) -> np.ndarray:
        """
        Apply a mask to an image based on green color in HSV space. 
        If the mask covers more than green_threshold of the image, apply the inverse of the mask.

        Args:
            image (np.ndarray): An image to apply the mask to.
            green_threshold (float): Threshold for green color coverage.

        Returns:
            np.ndarray: The masked image.
        """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        # Create the mask
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Count the number of masked pixels
        total_pixels = image.shape[0] * image.shape[1]
        masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
        mask_percentage = masked_pixels / total_pixels
        
        if mask_percentage > green_threshold:
            # Apply inverse mask
            return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        else:
            # Apply normal mask
            return image

    def clustering(self, img: np.ndarray) -> Tuple[int, int, int]:
        """
        Perform K-Means clustering on an image to identify the dominant jersey color.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Tuple[int, int, int]: The jersey color in BGR format.
        """
        # Reshape image to 2D array
        img_reshape = img.reshape(-1, 3)
        
        # K-Means clustering
        self.kmeans.fit(img_reshape)
        
        # Get Cluster Labels
        labels = self.kmeans.labels_
        
        # Reshape the labels into the image shape
        cluster_img = labels.reshape(img.shape[0], img.shape[1])

        # Get Jersey Color
        corners = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)

        # The other cluster is a player cluster
        player_cluster = 1 - bg_cluster

        jersey_color_bgr = self.kmeans.cluster_centers_[player_cluster]
        
        return (int(jersey_color_bgr[2]), int(jersey_color_bgr[1]), int(jersey_color_bgr[0]))

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
        # img_top = img[0:img.shape[0] // 2, :]  # Use upper half for jersey detection
        # masked_img_top = self.apply_mask(img_top, green_threshold=0.08)
        # jersey_color = self.clustering(masked_img_top)
        
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

    def assign_clubs(self, team_classifier,frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
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
                bbox = track['bbox']
                is_goalkeeper = (track_type == 'goalkeeper')
                cluster_label, _ = self.get_player_club(team_classifier,frame, bbox, player_id, is_goalkeeper)
                club = cluster_to_club.get(cluster_label, 'unknown')
                tracks[track_type][player_id]['club'] = club
                tracks[track_type][player_id]['club_color'] = self.club_colors[club]
        
        return tracks


# changes
    # def assign_clubs(self, frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
    #     tracks = tracks.copy()

    #     # Combine players and goalkeepers
    #     all_colors = []
    #     all_ids = []
    #     all_types = []

    #     for track_type in ['player', 'goalkeeper']:
    #         for player_id, track in tracks[track_type].items():
    #             bbox = track['bbox']
    #             jersey_color = self.get_jersey_color(frame, bbox, player_id)
    #             all_colors.append(jersey_color)
    #             all_ids.append(player_id)
    #             all_types.append(track_type)

    #     if len(all_colors) >= 4:
    #         # Perform K-Means clustering on all jersey colors with 4 clusters
    #         all_colors_np = np.array(all_colors)
    #         kmeans_all = KMeans(n_clusters=4, random_state=42)
    #         kmeans_all.fit(all_colors_np)
    #         labels_all = kmeans_all.labels_

    #         # Map clusters to clubs and roles
    #         centroids_all = kmeans_all.cluster_centers_

    #         # Known colors and their labels (club and role)
    #         known_colors = []
    #         known_labels = []
    #         for club_name in self.all_club_colors.keys():
    #             player_color = self.club_colors[club_name]
    #             goalkeeper_color = self.goalkeeper_colors[club_name]
    #             known_colors.append(player_color)
    #             known_labels.append((club_name, 'player'))
    #             known_colors.append(goalkeeper_color)
    #             known_labels.append((club_name, 'goalkeeper'))

    #         known_colors_np = np.array(known_colors)

    #         # Assign clusters to closest known colors
    #         cluster_to_club_role = {}
    #         for cluster_label in range(4):
    #             centroid = centroids_all[cluster_label]
    #             distances = np.linalg.norm(known_colors_np - centroid, axis=1)
    #             min_index = np.argmin(distances)
    #             club_name, role = known_labels[min_index]
    #             cluster_to_club_role[cluster_label] = (club_name, role)

    #         # Assign clubs and roles to all individuals
    #         for idx, player_id in enumerate(all_ids):
    #             cluster_label = labels_all[idx]
    #             club_name, role = cluster_to_club_role[cluster_label]
    #             track_type = all_types[idx]
    #             tracks[track_type][player_id]['club'] = club_name
    #             # Update the type based on clustering if necessary
    #             if role != track_type:
    #                 # Move the track to the correct type
    #                 tracks[role][player_id] = tracks[track_type].pop(player_id)
    #                 track_type = role
    #             # Assign the appropriate club color
    #             if track_type == 'goalkeeper':
    #                 tracks[track_type][player_id]['club_color'] = self.goalkeeper_colors[club_name]
    #             else:
    #                 tracks[track_type][player_id]['club_color'] = self.club_colors[club_name]
    #     else:
    #         # Not enough individuals to cluster, assign based on closest color
    #         for idx, player_id in enumerate(all_ids):
    #             color = all_colors[idx]
    #             track_type = all_types[idx]
    #             pred = self.model.predict(color, is_goalkeeper=(track_type == 'goalkeeper'))
    #             club_name = list(self.all_club_colors.keys())[pred]
    #             tracks[track_type][player_id]['club'] = club_name
    #             if track_type == 'goalkeeper':
    #                 tracks[track_type][player_id]['club_color'] = self.goalkeeper_colors[club_name]
    #             else:
    #                 tracks[track_type][player_id]['club_color'] = self.club_colors[club_name]

    #     return tracks



class ClubAssignerModel:
    def __init__(self, club1: Club, club2: Club) -> None:
        """
        Initializes the ClubAssignerModel with jersey colors for the clubs.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
        """
        self.player_centroids = np.array([club1.player_jersey_color, club2.player_jersey_color])
        self.goalkeeper_centroids = np.array([club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])

    def predict(self, extracted_color: Tuple[int, int, int], is_goalkeeper: bool = False) -> int:
        """
        Predict the club for a given jersey color based on the centroids.

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in BGR format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            int: The index of the predicted club (0 or 1).
        """
        if is_goalkeeper:
            centroids = self.goalkeeper_centroids
        else:
            centroids = self.player_centroids

        # Calculate distances
        distances = np.linalg.norm(extracted_color - centroids, axis=1)
        
        return np.argmin(distances)
