from abc import ABC, abstractmethod

class AbstractAnnotator(ABC):

    @abstractmethod
    def annotate(self, frame, tracks):
        """
        Abstract method for annotation
        
        Args:
            frame (array): The current frame for detection.
            tracks (dict): Tracks data
        
        Returns:
            Annotated frame
        """
        pass
        