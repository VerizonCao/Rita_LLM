#!/usr/bin/env python
# coding: utf-8
import queue
import threading


class ALRealtimeIOAndStatus:
    # --- Status Variables ---
    DATA_TRANSMISSION_STARTED = 1
    DATA_TRANSMISSION_COMPLETED = 0
    USER_TALKING = -1

    def __init__(
        self,
        audio_source_wavform_queue: queue.Queue = queue.Queue(),
        wav2vec_preprocessed_queue: queue.Queue = queue.Queue(),
        text_input_queue: queue.Queue = queue.Queue(),
    ):
        # across thread
        self._lock = threading.Lock()
        """
        voice-input-> wav2vec-cpu
        * feed into audio_source_wavform_queue, voice module 
        * set _is_voice_transmitting = True/False
        * set set_user_talking = True
        wav2vec-cpu -> wav2vec-device
        * feed into wav2vec_preprocessed_queue
        * set _is_model_speaking = True
        wav2vec-device & main thread
        * model will keep speaking until 
          1. user-talking is detected
          2. data-transmission is completed && 
             wav2vec_preprocessed_queue is empty &&
             wav2vec-cpu frame_index is about to exceed total_frames
             ( this is only checked in main thread )
        * set _is_model_speaking = False
        """
        self._is_model_speaking = False
        self._is_voice_transmitting = False
        self._curr_expression_keys = "neutral-neutral"
        self.portrait = None  # only used for setting expressions
        # input processor queues references
        self.audio_source_wavform_queue = (
            audio_source_wavform_queue  # voice to wav2vec-cpu
        )
        self.wav2vec_preprocessed_queue = (
            wav2vec_preprocessed_queue  # wav2vec-cpu to wav2vec-device
        )
        self.text_input_queue = text_input_queue  # text to motion

    def _flush_queue(self, queue: queue.Queue):
        with self._lock:
            while not queue.empty():
                queue.get_nowait()

    def _flush_audio_source_wavform_queue(self):
        self._flush_queue(self.audio_source_wavform_queue)

    def _flush_wav2vec_preprocessed_queue(self):
        self._flush_queue(self.wav2vec_preprocessed_queue)

    def flush_all_queues(self):
        self._flush_audio_source_wavform_queue()
        self._flush_wav2vec_preprocessed_queue()

    def set_voice_transmit_started(self):
        with self._lock:
            self._is_voice_transmitting = True

    def set_voice_transmit_completed(self):
        with self._lock:
            self._is_voice_transmitting = False

    def get_is_voice_transmitting(self):
        with self._lock:
            return self._is_voice_transmitting

    def set_model_stops_speaking(self):
        with self._lock:
            self._is_model_speaking = False

    def set_model_starts_speaking(self):
        with self._lock:
            self._is_model_speaking = True

    def get_is_model_speaking(self):
        with self._lock:
            return self._is_model_speaking

    def set_user_talking(self):
        # model instantly stop. Clear buffers.
        with self._lock:
            self._is_model_speaking = False
            self._is_voice_transmitting = False

    def set_curr_expression_keys(self, expression_keys: str):
        with self._lock:
            self._curr_expression_keys = expression_keys
            category, name = self._curr_expression_keys.split("-")
            if self.portrait:
                self.portrait.elapsed_select_expression_by_keys(category, name)
