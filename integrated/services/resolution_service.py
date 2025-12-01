from typing import Dict
from .resolution_methods import ResolutionMethod, RandomResolutionMethod


class ResolutionService:
    
    @staticmethod
    def get_all_resolution_methods() -> Dict[str, ResolutionMethod]:
        return {
            "Random": RandomResolutionMethod()
        }

