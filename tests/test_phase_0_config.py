from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase_0.config import load_config, validate_config

def test_phase_0_config():
    c = load_config(ROOT / "configs" / "phase_0.yaml")
    assert c["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert c["datasets"]["hindi"]["swap_train_test"] is True
    assert c["lora"]["rank"] == 16 and c["lora"]["alpha"] == 32
    assert set(c["lora"]["target_modules"]) == {"q_proj","k_proj","v_proj","o_proj"}

def test_missing_config_sections_raise():
    try:
        validate_config({})
    except ValueError as e:
        assert "Missing config sections" in str(e)
    else:
        raise AssertionError("Expected ValueError")
