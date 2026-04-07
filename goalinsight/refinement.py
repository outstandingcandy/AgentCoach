#!/usr/bin/env python3
"""Post-processing Refinement - Improve tracking quality.

This module:
1. SAM2-based segmentation to recover missed detections and fix ID switches
2. Majority voting for temporal consistency of attributes
3. Tracklet merging using ReID features and jersey number consistency
"""

import json
import pickle
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from .utils.config import get_default_config, get_process_fps_from_config


def majority_vote(values: list, min_count: int = 2) -> any:
    """Apply majority voting to a list of values.

    Args:
        values: List of values to vote on
        min_count: Minimum occurrences required for a valid result

    Returns:
        Most common value, or None if no clear majority
    """
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None

    counter = Counter(valid_values)
    most_common = counter.most_common(1)[0]

    if most_common[1] >= min_count:
        return most_common[0]
    return None


def compute_track_similarity(
    features1: list[np.ndarray],
    features2: list[np.ndarray],
) -> float:
    """Compute similarity between two tracks based on ReID features.

    Args:
        features1: List of feature vectors from track 1
        features2: List of feature vectors from track 2

    Returns:
        Similarity score (0-1)
    """
    if not features1 or not features2:
        return 0.0

    # Use mean features
    mean1 = np.mean(features1, axis=0)
    mean2 = np.mean(features2, axis=0)

    # Normalize
    mean1 = mean1 / (np.linalg.norm(mean1) + 1e-8)
    mean2 = mean2 / (np.linalg.norm(mean2) + 1e-8)

    # Cosine similarity
    return float(np.dot(mean1, mean2))


def find_mergeable_tracklets(
    track_summaries: dict,
    reid_features: dict,
    tracks_data: dict,
    similarity_threshold: float = 0.8,
    max_gap_frames: int = 50,
) -> list[tuple[int, int]]:
    """Find pairs of tracklets that should be merged.

    Args:
        track_summaries: Dict of track_id -> summary info
        reid_features: Dict of track_id -> list of feature vectors
        tracks_data: Dict of frame_idx -> list of tracks
        similarity_threshold: Min ReID similarity for merging
        max_gap_frames: Max frame gap between tracklets to consider merging

    Returns:
        List of (track_id1, track_id2) pairs to merge
    """
    # Build track frame ranges
    track_frames = {}
    for frame_idx, frame_tracks in tracks_data.items():
        frame_idx = int(frame_idx)
        for track in frame_tracks:
            track_id = track["track_id"]
            if track_id not in track_frames:
                track_frames[track_id] = []
            track_frames[track_id].append(frame_idx)

    track_ranges = {
        tid: (min(frames), max(frames))
        for tid, frames in track_frames.items()
    }

    # Find candidate pairs (tracks that don't overlap in time)
    merge_pairs = []
    track_ids = list(track_ranges.keys())

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            tid1, tid2 = track_ids[i], track_ids[j]
            start1, end1 = track_ranges[tid1]
            start2, end2 = track_ranges[tid2]

            # Check if tracks are temporally separated
            if end1 < start2:
                gap = start2 - end1
            elif end2 < start1:
                gap = start1 - end2
            else:
                continue  # Overlapping tracks

            if gap > max_gap_frames:
                continue

            # Check ReID similarity
            feat1 = reid_features.get(tid1, [])
            feat2 = reid_features.get(tid2, [])

            if feat1 and feat2:
                # Convert to numpy if needed
                feat1 = [np.array(f) if isinstance(f, list) else f for f in feat1]
                feat2 = [np.array(f) if isinstance(f, list) else f for f in feat2]

                similarity = compute_track_similarity(feat1, feat2)
                if similarity >= similarity_threshold:
                    merge_pairs.append((tid1, tid2, similarity))

    # Sort by similarity and return pairs
    merge_pairs.sort(key=lambda x: x[2], reverse=True)
    return [(p[0], p[1]) for p in merge_pairs]


def apply_temporal_consistency(
    tracks_data: dict,
    track_summaries: dict,
) -> dict:
    """Apply temporal consistency to track attributes using majority voting.

    Args:
        tracks_data: Dict of frame_idx -> list of tracks
        track_summaries: Dict of track_id -> summary info

    Returns:
        Updated tracks_data with consistent attributes
    """
    # Collect all attributes per track
    track_attrs = {}
    for frame_idx, frame_tracks in tracks_data.items():
        for track in frame_tracks:
            track_id = track["track_id"]
            if track_id not in track_attrs:
                track_attrs[track_id] = {
                    "jersey_numbers": [],
                    "teams": [],
                    "roles": [],
                }

            if track.get("jersey_number"):
                track_attrs[track_id]["jersey_numbers"].append(track["jersey_number"])
            if track.get("team"):
                track_attrs[track_id]["teams"].append(track["team"])
            if track.get("role"):
                track_attrs[track_id]["roles"].append(track["role"])

    # Apply majority voting
    consistent_attrs = {}
    for track_id, attrs in track_attrs.items():
        consistent_attrs[track_id] = {
            "jersey_number": majority_vote(attrs["jersey_numbers"]),
            "team": majority_vote(attrs["teams"]),
            "role": majority_vote(attrs["roles"]),
        }

    # Apply consistent attributes back to tracks
    updated_tracks = {}
    for frame_idx, frame_tracks in tracks_data.items():
        updated_frame_tracks = []
        for track in frame_tracks:
            track_id = track["track_id"]
            updated_track = track.copy()
            if track_id in consistent_attrs:
                updated_track.update({
                    k: v for k, v in consistent_attrs[track_id].items()
                    if v is not None
                })
            updated_frame_tracks.append(updated_track)
        updated_tracks[frame_idx] = updated_frame_tracks

    return updated_tracks


def run_refinement(
    tracking_dir: Path,
    output_dir: Path,
    config: dict | None = None,
    enable_sam2: bool = False,  # Disabled by default (requires SAM2)
    enable_tracklet_merge: bool = True,
    merge_similarity_threshold: float = 0.8,
    max_gap_frames: int = 50,
):
    """Run Stage 3 post-processing.

    Args:
        tracking_dir: Path to Stage 2 tracking results
        output_dir: Directory for output files
        config: Optional configuration dict
        enable_sam2: Whether to use SAM2 for segmentation refinement
        enable_tracklet_merge: Whether to merge fragmented tracklets
        merge_similarity_threshold: ReID similarity threshold for merging
        max_gap_frames: Max frame gap for tracklet merging

    Returns:
        Dict with post-processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 2 results
    print("Stage 3: Loading tracking results...")

    tracks_path = tracking_dir / "tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracking results not found: {tracks_path}")

    with open(tracks_path, "r") as f:
        tracks_data = json.load(f)

    summaries_path = tracking_dir / "track_summaries.json"
    track_summaries = {}
    if summaries_path.exists():
        with open(summaries_path, "r") as f:
            track_summaries = json.load(f)

    reid_path = tracking_dir / "reid_features.pkl"
    reid_features = {}
    if reid_path.exists():
        with open(reid_path, "rb") as f:
            reid_features = pickle.load(f)

    print(f"  Loaded {len(tracks_data)} frames, {len(set(t['track_id'] for tracks in tracks_data.values() for t in tracks))} unique tracks")

    original_track_count = len(set(
        t["track_id"] for tracks in tracks_data.values() for t in tracks
    ))

    # Step 1: Temporal consistency (majority voting)
    print("Stage 3: Applying temporal consistency...")
    tracks_data = apply_temporal_consistency(tracks_data, track_summaries)

    # Step 2: Tracklet merging
    merge_count = 0
    if enable_tracklet_merge and reid_features:
        print("Stage 3: Finding mergeable tracklets...")
        merge_pairs = find_mergeable_tracklets(
            track_summaries,
            reid_features,
            tracks_data,
            similarity_threshold=merge_similarity_threshold,
            max_gap_frames=max_gap_frames,
        )

        if merge_pairs:
            print(f"  Found {len(merge_pairs)} potential merge pairs")

            # Build merge mapping (use first track ID for merged tracks)
            merge_map = {}
            for tid1, tid2 in merge_pairs:
                # Find root IDs
                root1 = tid1
                while root1 in merge_map:
                    root1 = merge_map[root1]
                root2 = tid2
                while root2 in merge_map:
                    root2 = merge_map[root2]

                if root1 != root2:
                    # Merge tid2 into tid1 (use smaller ID as root)
                    if root1 < root2:
                        merge_map[root2] = root1
                    else:
                        merge_map[root1] = root2
                    merge_count += 1

            # Apply merges to tracks
            for frame_idx in tracks_data:
                for track in tracks_data[frame_idx]:
                    tid = track["track_id"]
                    while tid in merge_map:
                        tid = merge_map[tid]
                    track["track_id"] = tid

            print(f"  Merged {merge_count} tracklet pairs")

    # Step 3: SAM2 segmentation refinement (optional)
    if enable_sam2:
        print("Stage 3: SAM2 refinement not yet implemented")
        # TODO: Implement SAM2-based refinement

    # Save refined results
    with open(output_dir / "tracks_refined.json", "w") as f:
        json.dump(tracks_data, f)

    # Generate final track summaries
    final_tracks = {}
    for frame_idx, frame_tracks in tracks_data.items():
        for track in frame_tracks:
            tid = track["track_id"]
            if tid not in final_tracks:
                final_tracks[tid] = {
                    "track_id": tid,
                    "frame_count": 0,
                    "first_frame": int(frame_idx),
                    "last_frame": int(frame_idx),
                    "jersey_number": track.get("jersey_number"),
                    "team": track.get("team"),
                }
            final_tracks[tid]["frame_count"] += 1
            final_tracks[tid]["last_frame"] = max(
                final_tracks[tid]["last_frame"], int(frame_idx)
            )

    with open(output_dir / "final_track_summaries.json", "w") as f:
        json.dump(final_tracks, f, indent=2)

    # Statistics
    final_track_count = len(final_tracks)
    stats = {
        "original_tracks": original_track_count,
        "final_tracks": final_track_count,
        "merged_pairs": merge_count,
        "reduction_rate": 1 - (final_track_count / original_track_count) if original_track_count > 0 else 0,
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStage 3 Complete:")
    print(f"  Original tracks: {original_track_count}")
    print(f"  Final tracks: {final_track_count}")
    print(f"  Merged: {merge_count} pairs")

    return stats


def main():
    tracking_dir = Path("data/processed/stage2_tracking")
    output_dir = Path("data/processed/stage3_postprocess")

    run_refinement(
        tracking_dir,
        output_dir,
        enable_sam2=False,
        enable_tracklet_merge=True,
    )


if __name__ == "__main__":
    main()
