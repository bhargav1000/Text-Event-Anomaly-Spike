from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

class EventAnomalyDetector:
    def __init__(self, window_size=20, max_features=2**12, iforest_kwargs=None):
        self.window_size = window_size
        self.vec = HashingVectorizer(n_features=max_features,
                                     alternate_sign=False,
                                     analyzer="word",
                                     ngram_range=(1,2))
        self.iforest = IsolationForest(contamination="auto", random_state=42, **(iforest_kwargs or {}))

    def _windows(self, events):
        for i in range(len(events) - self.window_size + 1):
            yield i, events[i:i+self.window_size]

    def fit(self, normal_events):
        X = [" ".join(win) for _, win in self._windows(normal_events)]
        Xv = self.vec.transform(X)
        self.iforest.fit(Xv)
        return self

    def score(self, events):
        out = []
        for i, win in self._windows(events):
            text = " ".join(win)
            Xv = self.vec.transform([text])
            score = -self.iforest.decision_function(Xv)[0]  # higher = more anomalous
            reason = self._rule_checks(win)
            out.append({"start_idx": i, "end_idx": i+self.window_size-1,
                        "score": float(score), "rule_reason": reason, "window": win})
        return out

    def _rule_checks(self, window):
        # simple illustrative rules
        joined = " ".join(window)
        if "botdead" in joined and "receiveddamage" in joined:
            # crude: if 'receiveddamage' appears after 'botdead'
            dead_idx = joined.find("botdead")
            dmg_idx = joined.find("receiveddamage")
            if dmg_idx > dead_idx:
                return "Damage after BotDead"
        if "reward_granted" in joined and "objective_advance" not in joined:
            return "Reward before objective advanced"
        return None
