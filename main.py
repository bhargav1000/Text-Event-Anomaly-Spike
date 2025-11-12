from detector import EventAnomalyDetector

# synthetic logs: mostly normal flow
normal = []
for lvl in range(5):
    normal += [
        "session_start",
        "level_start id={}".format(lvl),
        "playerspawned id=1",
        "receiveddamage src=bot42 amount=3",
        "objective_advance name=reach_gate",
        "reward_granted name=checkpoint",
        "botdead id=42",
        "level_end id={}".format(lvl)
    ]

# test stream with a couple anomalies
test_stream = normal[:]
test_stream += [
    # anomaly 1: damage after death
    "playerspawned id=2",
    "botdead id=99",
    "receiveddamage src=bot99 amount=5",
    # anomaly 2: reward before objective advanced
    "level_start id=99",
    "reward_granted name=treasure",
    "playerspawned id=3",
]

det = EventAnomalyDetector(window_size=8).fit(normal)
scores = det.score(test_stream)

# show top flags
scores = sorted(scores, key=lambda d: (d["rule_reason"] is None, -d["score"]))[:5]
for s in scores:
    print(f"[{s['start_idx']}..{s['end_idx']}] score={s['score']:.3f} rule={s['rule_reason']}")
    for e in s["window"]:
        print("  -", e)
    print()
