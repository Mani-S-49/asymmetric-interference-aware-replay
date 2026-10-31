from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase_0.metrics import compute_metrics, forgetting

def test_forgetting_when_accuracy_drops():
    assert abs(forgetting(.80,.72)-.08) < 1e-9

def test_forgetting_when_accuracy_improves():
    assert abs(forgetting(.70,.75)-(-.05)) < 1e-9

def test_binary_metrics():
    logits = np.array([[3.,0.],[0.,2.],[0.,1.],[4.,0.]])
    labels = np.array([0,1,1,0])
    m = compute_metrics((logits,labels))
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0
