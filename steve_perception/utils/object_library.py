"""Taught objects (object library): which of them a segmented object is, from its appearance fingerprints.

A taught object is a target profile YAML (steve_wbc config/targets/<name>.yaml, written by steve_wbc's
object_teacher) plus <name>_views.npz next to it: the SAM fingerprints of its views ("emb", N x 256, L2-normalised,
see model_server gsam embed) and their colour signatures ("sig", N x appearance.SIZE). The profile also holds the
fingerprint threshold calibrated from its own views ("match_threshold").

    lib = ObjectLibrary.load(targets_dir)            # every profile that has views
    m = lib.match(embedding, "white_mug")            # -> Match(score, other, other_score, ok, why)

An object counts as the target only when it is BOTH like the target (score >= its threshold) AND clearly more
like it than like any other taught object (score - other_score >= margin): a look-alike that was taught too is
refused rather than picked. Pure NumPy + PyYAML (runs in the system Python of the detector and the teacher).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MARGIN = 0.03          # the target must beat every other taught object by this much (cosine similarity)
THRESH_SLACK = 0.03    # threshold = the views' leave-one-out similarity minimum minus this
THRESH_RANGE = (0.80, 0.95)  # never below (anything small looks ~0.8 alike) / above (no view would pass)


def views_path(profile_path, profile=None):
    """<name>_views.npz of a profile YAML (or the file its 'views' entry names)."""
    p = Path(profile_path)
    prof = profile or {}
    name = prof.get("views") or (prof.get("extra") or {}).get("views") or f"{p.stem}_views.npz"
    return p.parent / name


def calibrate_threshold(emb):
    """Fingerprint threshold from an object's own views (N x 256): how alike each view is to the nearest OTHER view
    (leave-one-out), the lowest of those minus THRESH_SLACK, clipped to THRESH_RANGE. One view: the range's low end."""
    E = np.asarray(emb, float)
    if len(E) < 2:
        return THRESH_RANGE[0], []
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    loo = S.max(axis=1)
    return float(np.clip(loo.min() - THRESH_SLACK, *THRESH_RANGE)), [float(v) for v in loo]


@dataclass
class Match:
    score: float          # best similarity to one of the target's views
    other: str            # the most alike OTHER taught object ("" if none)
    other_score: float    # ... its best similarity
    ok: bool
    why: str              # reason when not ok


class ObjectLibrary:
    def __init__(self, objects):
        self.objects = objects  # name -> {"emb": N x 256, "sig": N x S | None, "threshold": float, "profile": dict}

    @classmethod
    def load(cls, folder):
        import yaml
        objects = {}
        for y in sorted(Path(folder).glob("*.yaml")):
            try:
                prof = yaml.safe_load(y.read_text()) or {}
            except Exception:
                continue
            v = views_path(y, prof)
            if not v.is_file():
                continue
            d = np.load(v)
            emb = np.asarray(d["emb"], float)
            if emb.ndim != 2 or not len(emb):
                continue
            objects[prof.get("name", y.stem)] = {
                "emb": emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9),
                "sig": np.asarray(d["sig"], float) if "sig" in d else None,
                "threshold": float(prof.get("match_threshold") or (prof.get("extra") or {}).get("match_threshold")
                                   or calibrate_threshold(emb)[0]),
                "profile": prof}
        return cls(objects)

    def names(self):
        return list(self.objects)

    def score(self, emb, name):
        e = np.asarray(emb, float)
        e = e / max(np.linalg.norm(e), 1e-9)
        return float(np.max(self.objects[name]["emb"] @ e))

    def match(self, emb, target, margin=MARGIN):
        if target not in self.objects:
            return Match(0.0, "", 0.0, False, f"'{target}' has no taught views")
        if emb is None:
            return Match(0.0, "", 0.0, False, "no fingerprint")
        s = self.score(emb, target)
        others = [(self.score(emb, n), n) for n in self.objects if n != target]
        so, on = max(others) if others else (0.0, "")
        thr = self.objects[target]["threshold"]
        if s < thr:
            return Match(s, on, so, False, f"not {target} ({s:.2f} < {thr:.2f})")
        if others and s - so < margin:
            return Match(s, on, so, False, f"as much like {on} ({s:.2f} vs {so:.2f})")
        return Match(s, on, so, True, "")
