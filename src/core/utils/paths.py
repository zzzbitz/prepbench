from pathlib import Path
import re
from config.experiment_config import ExperimentConfig

def get_output_path(case_path: Path, config: ExperimentConfig) -> Path:
    """
    Calculate the output directory for a given case and configuration.
    
    Args:
        case_path: Path to the case directory (e.g., data/case_001).
        config: The experiment configuration object.
        
    Returns:
        Path object representing the absolute path to the output directory.
    """
    model_name = config.model_name
    
    # Create a sanitized model_info for file paths
    model_info_base = model_name.split("/")[-1]
    model_info = re.sub(r"[^\w\-.:]", "_", model_info_base)
    
    case_name = case_path.name
    
    output_root = Path(config.output_root_template.format(
        model_info=model_info,
        run_mode=config.run_mode,
        case_name=case_name
    )).resolve()
    
    return output_root
