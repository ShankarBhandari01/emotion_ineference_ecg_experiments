import numpy as np
import scipy.io
import neurokit2 as nk
import logging
from typing import Tuple, List, Union

logger = logging.getLogger("ml.preprocessing.DreamerECGLoader")
logging.basicConfig(level=logging.INFO)


class DreamerECGLoader:
    def __init__(self, dreamer_data_path: str):
        self.dreamer_data_path = dreamer_data_path

    def ensure_array(self, obj):
        return obj if isinstance(obj, (list, np.ndarray)) else [obj]

    def load_dreamer_ecg_segments(
            self,
            segment_length_sec: float = 10.0,
            overlap_ratio: float = 0.0,
            return_arrays: bool = True,
            min_segment_length_sec: float = 5.0,
            ecg_channel: int = 0
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[List[float]]]]:
        try:
            mat = scipy.io.loadmat(self.dreamer_data_path, struct_as_record=False, squeeze_me=True)
            dreamer = mat["DREAMER"]
            subjects = self.ensure_array(dreamer.Data)
            ecg_sampling_rate = dreamer.ECG_SamplingRate
            n_videos = dreamer.noOfVideoSequences

            segment_len = int(segment_length_sec * ecg_sampling_rate)
            min_segment_len = int(min_segment_length_sec * ecg_sampling_rate)
            step_size = int(segment_len * (1 - overlap_ratio))

            all_segments, all_labels = [], []

            for subj_idx, subj in enumerate(subjects):
                ecg_data = subj.ECG
                stimuli_list = self.ensure_array(ecg_data.stimuli)
                valence = self.ensure_array(subj.ScoreValence)
                arousal = self.ensure_array(subj.ScoreArousal)
                dominance = self.ensure_array(subj.ScoreDominance)

                if len(stimuli_list) != n_videos:
                    logger.warning(f"Subject {subj_idx}: expected 18 stimuli, got {len(stimuli_list)}")
                    continue

                for video_idx in range(n_videos):
                    try:
                        raw_matrix = stimuli_list[video_idx]  # shape: (M, 2)
                        if raw_matrix is None or raw_matrix.shape[0] < min_segment_len:
                            logger.warning(f"Subject {subj_idx}, video {video_idx}: ECG too short")
                            continue

                        ecg_raw = raw_matrix[:, ecg_channel]  # Select ECG channel 0
                        ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=ecg_sampling_rate)
                        ecg_norm = (ecg_clean - np.mean(ecg_clean)) / np.std(ecg_clean)

                        for start in range(0, len(ecg_norm) - segment_len + 1, step_size):
                            segment = ecg_norm[start:start + segment_len]
                            if not np.isnan(segment).any():
                                all_segments.append(segment)
                                all_labels.append([
                                    valence[video_idx],
                                    arousal[video_idx],
                                    dominance[video_idx]
                                ])
                    except Exception as e:
                        logger.warning(f"Subject {subj_idx}, video {video_idx} failed: {e}")
                        continue

            logger.info(f"Extracted {len(all_segments)} segments from DREAMER")

            return (np.array(all_segments), np.array(all_labels)) if return_arrays else (all_segments, all_labels)

        except Exception as e:
            logger.error(f"Failed to load DREAMER dataset: {e}")
            raise
