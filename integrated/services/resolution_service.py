from typing import Dict
from .resolution_methods import ResolutionMethod, DummyResolutionMethod, KociembaResolutionMethod


class ResolutionService:
    
    @staticmethod
    def get_all_resolution_methods() -> Dict[str, ResolutionMethod]:
        return {
            "Kociemba": KociembaResolutionMethod(),
            "Beginner Solver": DummyResolutionMethod()
        }

