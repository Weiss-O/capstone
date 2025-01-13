from abc import ABC, abstractmethod
import cv2

class ProposalPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image) -> list:
        pass

#Basic Proposal Preprocessor Class
class NullPreprocessor(ProposalPreprocessor):
    def preprocess(self, image) -> list:
        return image
    
class GrayscaleGaussianPreprocessor(ProposalPreprocessor):
    def preprocess(self, image) -> list:
        return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)