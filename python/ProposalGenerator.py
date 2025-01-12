from abc import ABC, abstractmethod

class ProposalGenerator(ABC):
    @abstractmethod
    def generateProposals(self, image, baseline) -> list:
        pass

#Basic Proposal Generator Class Using SSIM
class SSIMProposalGenerator(ProposalGenerator):
    def generateProposals(self, image, baseline) -> list:
        pass