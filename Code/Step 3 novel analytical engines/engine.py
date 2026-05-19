import torch
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ValuescopeAnalyticalEngines:
    """
    Core mathematical engines utilizing the 19-dimensional Schwartz Value Circular Continuum.
    """
    def __init__(self):
        self.num_values = 19
        self.angles = np.linspace(0, 2 * np.pi, self.num_values, endpoint=False)
        self.cos_theta = np.cos(self.angles)
        self.sin_theta = np.sin(self.angles)

    def _generate_2d_vector(self, probabilities: np.ndarray) -> np.ndarray:
        x_coord = np.sum(probabilities * self.cos_theta)
        y_coord = np.sum(probabilities * self.sin_theta)
        return np.array([x_coord, y_coord])

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return 0.0 if norm_product == 0 else dot_product / norm_product

    def calculate_dissonance_score(self, user_probs: np.ndarray, feed_probs: np.ndarray) -> float:
        v_user = self._generate_2d_vector(user_probs)
        v_feed = self._generate_2d_vector(feed_probs)
        return 1.0 - self._cosine_similarity(v_user, v_feed)

    def calculate_value_drift(self, base_probs: np.ndarray, curr_probs: np.ndarray) -> Dict[str, float]:
        v_base = self._generate_2d_vector(base_probs)
        v_curr = self._generate_2d_vector(curr_probs)
        
        v_drift = v_curr - v_base
        delta_x, delta_y = v_drift[0], v_drift[1]
        
        magnitude = np.sqrt(delta_x**2 + delta_y**2)
        direction_rad = np.arctan2(delta_y, delta_x) 
        
        return {
            "magnitude": magnitude,
            "direction_radians": direction_rad
        }

    def calculate_ethical_density(self, feed_batch_probs: np.ndarray) -> float:
        p_feed_aggregate = np.mean(feed_batch_probs, axis=0)
        v_feed_aggregate = self._generate_2d_vector(p_feed_aggregate)
        r_feed = np.linalg.norm(v_feed_aggregate)
        return max(0.0, 1.0 - r_feed)


class ValuescopePipeline:
    def __init__(self, model_path: str = "roberta-base"):
        print("Loading Valuescope Distilled Student Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=38, problem_type="multi_label_classification"
        )
        self.model.eval() 
        self.engine = ValuescopeAnalyticalEngines()

    def extract_19_base_probabilities(self, text: str) -> np.ndarray:
        """Translates 38 polarities into the 19 Schwartz Circle coordinates."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs_38 = torch.sigmoid(outputs.logits).squeeze().numpy()
        probs_19 = np.zeros(19)
        for i in range(19):
            probs_19[i] = probs_38[i*2] + probs_38[(i*2) + 1]
            
        return probs_19

    def execute_full_diagnostic(self, past_history: List[str], recent_history: List[str], feed: List[str]):
        """Executes all three patent claim engines."""
        # Process NLP inputs
        past_probs = np.array([self.extract_19_base_probabilities(t) for t in past_history])
        recent_probs = np.array([self.extract_19_base_probabilities(t) for t in recent_history])
        feed_probs = np.array([self.extract_19_base_probabilities(t) for t in feed])
        
        agg_past = np.mean(past_probs, axis=0)
        agg_recent = np.mean(recent_probs, axis=0)
        agg_feed = np.mean(feed_probs, axis=0)

        # Run Mathematical Engines
        dissonance = self.engine.calculate_dissonance_score(agg_recent, agg_feed)
        drift = self.engine.calculate_value_drift(agg_past, agg_recent)
        density = self.engine.calculate_ethical_density(feed_probs)

        return dissonance, drift, density

if __name__ == "__main__":
    valuescope = ValuescopePipeline(model_path="roberta-base")
    
    # Simulating data from 30 days ago (Tbase)
    mock_past_history = [
        "I loved helping out at the charity drive.",
        "Nature is so beautiful, we must protect it."
    ]
    
    # Simulating data from the last 7 days (Tcurr)
    mock_recent_history = [
        "I am so grateful for the community support today.",
        "It is important to respect the traditions of our elders."
    ]
    
    # Simulating the algorithmic feed (Vfeed)
    mock_algorithmic_feed = [
        "Crush your competition and buy this luxury car!",
        "You need to dominate your industry to be successful."
    ]
    
    # Execute the engines
    dissonance_score, value_drift, ethical_density = valuescope.execute_full_diagnostic(
        past_history=mock_past_history,
        recent_history=mock_recent_history, 
        feed=mock_algorithmic_feed
    )
    
    # Output Final Metrics
    print("\n" + "="*45)
    print("VALUESCOPE REAL-TIME DIAGNOSTIC REPORT")
    print("="*45)
    print(f"1. Value Dissonance Score : {dissonance_score:.4f} / 2.0")
    print(f"2. Drift Magnitude        : {value_drift['magnitude']:.4f}")
    print(f"3. Ethical Density Score  : {ethical_density:.4f} / 1.0")
    print("="*45)