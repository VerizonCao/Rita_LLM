import wave
import logging
from typing import Optional
from abc import ABC, abstractmethod

import pyaudio
import numpy as np
import asyncio
import threading
import time
import os

from util.user_audio_input.vad import VoiceActivityDetector, SileroVoiceActivityDetector
from livekit import rtc
from livekit.agents import utils

# Constants for PyAudio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # Default sample rate
FRAMES_PER_BUFFER = 1024

logger = logging.getLogger(__name__)


class AudioCaptureEventHandler(ABC):
    """
    Abstract base class defining the interface for handling audio capture events.
    Any event handler must implement these methods.
    """

    @abstractmethod
    def send_audio_data(self, audio_data: bytes):
        """
        Called to send audio data to the client.

        :param audio_data: Raw audio data in bytes.
        """
        pass

    @abstractmethod
    def on_speech_start(self):
        """
        Called when speech starts.
        """
        pass

    @abstractmethod
    def on_speech_end(self):
        """
        Called when speech ends.
        """
        pass

    @abstractmethod
    def on_keyword_detected(self, result):
        """
        Called when a keyword is detected.

        :param result: The recognition result containing details about the detected keyword.
        """
        pass


class AudioCaptureLiveKit:
    """
    Handles audio input processing, including Voice Activity Detection (VAD)
    and wave file handling using PyAudio. It communicates with an event handler
    to notify about audio data and speech events.
    """

    def __init__(
        self,
        event_handler: AudioCaptureEventHandler,
        audio_track: rtc.AudioTrack,
        sample_rate: int = RATE,
        channels: int = CHANNELS,
        frames_per_buffer: int = FRAMES_PER_BUFFER,
        buffer_duration_sec: float = 1.0,
        cross_fade_duration_ms: int = 20,
        vad_parameters: Optional[dict] = None,
        enable_wave_capture: bool = False,
        keyword_model_file: Optional[str] = None,
    ):
        """
        Initializes the AudioCapture instance.

        :param event_handler: An instance of AudioCaptureEventHandler to handle callbacks.
        :param audio_track: LiveKit audio track for capturing audio.
        :param sample_rate: Sampling rate for audio capture.
        :param channels: Number of audio channels.
        :param frames_per_buffer: Number of frames per buffer.
        :param buffer_duration_sec: Duration of the internal audio buffer in seconds.
        :param cross_fade_duration_ms: Duration for cross-fading in milliseconds.
        :param vad_parameters: Parameters for VoiceActivityDetector.
        :param enable_wave_capture: Flag to enable wave file capture.
        :param keyword_model_file: Path to the keyword recognition model file.
        """
        self.event_handler = event_handler
        self.audio_track = audio_track
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.cross_fade_duration_ms = cross_fade_duration_ms
        self.enable_wave_capture = enable_wave_capture
        self.wave_file = None
        self.wave_filename = None

        self.vad = None
        self.speech_started = False

        self.audio_task = None
        self.stream = None

        if vad_parameters is not None:
            try:
                if (
                    "model_path" in vad_parameters
                    and isinstance(vad_parameters["model_path"], str)
                    and vad_parameters["model_path"].strip()
                ):
                    self.vad = SileroVoiceActivityDetector(**vad_parameters)
                else:
                    self.vad = VoiceActivityDetector(**vad_parameters)
                logger.info(f"VAD module initialized with parameters: {vad_parameters}")
                self.buffer_duration_sec = buffer_duration_sec
                self.buffer_size = int(self.buffer_duration_sec * self.sample_rate)
                self.audio_buffer = np.zeros(self.buffer_size, dtype=np.int16)
                self.buffer_pointer = 0
                self.cross_fade_samples = int(
                    (self.cross_fade_duration_ms / 1000) * self.sample_rate
                )
            except Exception as e:
                logger.error(f"Failed to initialize VAD module: {e}")
                self.vad = None

        self.keyword_recognizer = None
        if False:
            try:
                self.keyword_recognizer = AzureKeywordRecognizer(
                    model_file=keyword_model_file,
                    callback=self._on_keyword_detected,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )
                logger.info("Keyword recognizer initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize AzureKeywordRecognizer: {e}")

        self.is_running = False

    def start(self):
        """
        Starts the audio capture stream and initializes necessary components.
        """
        if self.is_running:
            logger.warning("AudioCapture is already running.")
            return

        if self.enable_wave_capture:
            try:
                # Use a unique filename with timestamp
                timestamp = int(time.time())
                self.wave_filename = f"microphone_output_{timestamp}.wav"
                self.wave_file = wave.open(self.wave_filename, "wb")
                self.wave_file.setnchannels(self.channels) 
                self.wave_file.setsampwidth(2)  # 16-bit audio
                self.wave_file.setframerate(self.sample_rate)
                logger.info(f"Wave file initialized for capture: {self.wave_filename}")
            except Exception as e:
                logger.error(f"Error opening wave file: {e}")
                self.enable_wave_capture = False
                self.wave_file = None
                self.wave_filename = None

        if self.keyword_recognizer:
            try:
                self.keyword_recognizer.start_recognition()
                logger.info("Keyword recognizer started.")
            except Exception as e:
                logger.error(f"Failed to start AzureKeywordRecognizer: {e}")

        try:
            self.is_running = True
            loop = asyncio.new_event_loop()
            self.audio_task = loop.create_task(self._process_audio())
            logger.info("Audio stream started.")
            print("before the loop run forever !!")
            try:
                loop.run_forever()
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Failed to initialize RTC AudioTrack Async Loop: {e}")
            self.is_running = False
            raise

    def on_text_received(self, text: str):
        # when user direct type text as the input, now we just trigger the agent response.
        self.event_handler.on_text_received(text)

    def stop(self, terminate: bool = False):
        """
        Stops the audio capture stream and releases all resources.
        """
        if not self.is_running:
            logger.warning("AudioCapture is already stopped.")
            return

        try:
            if self.audio_task:
                self.audio_task.cancel()
                logger.info("Audio processing task canceled.")
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")

        if self.keyword_recognizer:
            try:
                self.keyword_recognizer.stop_recognition()
                logger.info("Keyword recognizer stopped.")
            except Exception as e:
                logger.error(f"Error stopping AzureKeywordRecognizer: {e}")

        if self.enable_wave_capture and self.wave_file:
            try:
                self.wave_file.close()
                logger.info(f"Wave file saved successfully: {self.wave_filename}")
                # Clean up the wave file
                if self.wave_filename and os.path.exists(self.wave_filename):
                    os.remove(self.wave_filename)
                    logger.info(f"Removed wave file: {self.wave_filename}")
            except Exception as e:
                logger.error(f"Error closing wave file: {e}")
            finally:
                self.wave_file = None
                self.wave_filename = None

        self.is_running = False
        logger.info("AudioCapture has been stopped.")

    async def _process_audio(self):
        """Coroutine to continuously process audio from rtc.AudioTrack."""
        logger.info("Starting audio processing...")
        print("in process_audio function !!")

        # Calculate buffer sizes based on parameters
        bytes_per_sample = 2  # For FORMAT = pyaudio.paInt16
        samples_per_frame = self.frames_per_buffer  # Number of samples per frame
        frame_size = samples_per_frame * bytes_per_sample * self.channels
        buffer = bytearray()

        while self.is_running:
            # Wait until audio track is available
            while self.audio_track is None and self.is_running:

                # we also handle it here
                if self.event_handler and self.event_handler.has_text_input():
                    self.event_handler.on_text_handled()

                await asyncio.sleep(0.1)

            if not self.is_running:
                break

            logger.info(
                f"Audio track status: {'Available' if self.audio_track else 'None'}"
            )

            # Initialize audio stream
            audio_stream = rtc.AudioStream(
                self.audio_track, sample_rate=self.sample_rate, channels=self.channels
            )
            cur_track = self.audio_track

            try:
                async for event in audio_stream:
                    if not self.is_running:
                        break

                    # Check if track has changed
                    if self.audio_track != cur_track:
                        logger.info(
                            f"Audio track changed from {cur_track} to {self.audio_track}, creating new stream..."
                        )
                        break

                    frame: rtc.AudioFrame = event.frame
                    buffer.extend(bytes(frame.data))

                    # Process complete frames when we have enough data
                    while len(buffer) >= frame_size:
                        # Extract one frame worth of data
                        frame_data = buffer[:frame_size]
                        buffer = buffer[frame_size:]

                        # Process the frame
                        self._handle_audio_data(bytes(frame_data))

            except Exception as e:
                logger.error(f"Error in audio stream processing: {e}")
                await asyncio.sleep(0.1)  # Small delay before retrying
                continue

            # If we get here, either the track changed or there was an error
            # Clear the buffer when switching tracks
            buffer.clear()

    # new function
    def _handle_audio_data(self, indata: bytes):

        # we check text data first, if we have text data, we handle it first.
        if self.event_handler and self.event_handler.has_text_input():
            print("has text, continue")
            self.event_handler.on_text_handled()
            return

        try:
            audio_data = np.frombuffer(indata, dtype=np.int16).copy()
            # print(f"audio data {audio_data[:10]}")
        except ValueError as e:
            logger.error(f"Error converting audio data: {e}")
            return (None, pyaudio.paContinue)

        if self.vad is None:
            self.event_handler.send_audio_data(indata)
            if self.enable_wave_capture and self.wave_file:
                try:
                    self.wave_file.writeframes(indata)
                except Exception as e:
                    logger.error(f"Error writing to wave file: {e}")
            return (None, pyaudio.paContinue)

        try:
            speech_detected, is_speech = self.vad.process_audio_chunk(audio_data)
            if speech_detected:
                print("speech_detected: ", speech_detected)
            if is_speech:
                print("is_speech: ", is_speech)
            if self.keyword_recognizer:
                self.keyword_recognizer.push_audio(audio_data)
        except Exception as e:
            logger.error(f"Error processing VAD: {e}")
            speech_detected, is_speech = False, False

        if speech_detected or self.speech_started:
            if is_speech:
                if not self.speech_started:
                    logger.info("Speech started")
                    self.buffer_pointer = self._update_buffer(
                        audio_data,
                        self.audio_buffer,
                        self.buffer_pointer,
                        self.buffer_size,
                    )
                    current_buffer = self._get_buffer_content(
                        self.audio_buffer, self.buffer_pointer, self.buffer_size
                    ).copy()

                    fade_length = min(
                        self.cross_fade_samples, len(current_buffer), len(audio_data)
                    )
                    if fade_length > 0:
                        fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
                        fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)

                        buffer_fade_section = current_buffer[-fade_length:].astype(
                            np.float32
                        )
                        audio_fade_section = audio_data[:fade_length].astype(np.float32)

                        faded_buffer_section = buffer_fade_section * fade_out
                        faded_audio_section = audio_fade_section * fade_in

                        current_buffer[-fade_length:] = np.round(
                            faded_buffer_section
                        ).astype(np.int16)
                        audio_data[:fade_length] = np.round(faded_audio_section).astype(
                            np.int16
                        )

                        combined_audio = np.concatenate((current_buffer, audio_data))
                    else:
                        combined_audio = audio_data

                    logger.info("Sending buffered audio to client via event handler...")
                    self.event_handler.on_speech_start()
                    self.event_handler.send_audio_data(combined_audio.tobytes())
                    if self.enable_wave_capture and self.wave_file:
                        try:
                            self.wave_file.writeframes(combined_audio.tobytes())
                        except Exception as e:
                            logger.error(f"Error writing to wave file: {e}")
                else:
                    self.event_handler.send_audio_data(audio_data.tobytes())
                    if self.enable_wave_capture and self.wave_file:
                        try:
                            self.wave_file.writeframes(audio_data.tobytes())
                        except Exception as e:
                            logger.error(f"Error writing to wave file: {e}")
                self.speech_started = True
            else:
                logger.info("Speech ended")
                self.event_handler.on_speech_end()
                self.speech_started = False

        if self.vad:
            self.buffer_pointer = self._update_buffer(
                audio_data, self.audio_buffer, self.buffer_pointer, self.buffer_size
            )

    def handle_input_audio(self, indata: bytes, frame_count: int, time_info, status):
        """
        Combined callback function for PyAudio input stream.
        Processes incoming audio data, performs VAD, and triggers event handler callbacks.

        :param indata: Incoming audio data in bytes.
        :param frame_count: Number of frames.
        :param time_info: Time information.
        :param status: Status flags.
        :return: Tuple containing None and pyaudio.paContinue.
        """
        if status:
            logger.warning(f"Input Stream Status: {status}")

        try:
            audio_data = np.frombuffer(indata, dtype=np.int16).copy()
        except ValueError as e:
            logger.error(f"Error converting audio data: {e}")
            return (None, pyaudio.paContinue)

        if self.vad is None:
            self.event_handler.send_audio_data(indata)
            if self.enable_wave_capture and self.wave_file:
                try:
                    self.wave_file.writeframes(indata)
                except Exception as e:
                    logger.error(f"Error writing to wave file: {e}")
            return (None, pyaudio.paContinue)

        try:
            speech_detected, is_speech = self.vad.process_audio_chunk(audio_data)
            if self.keyword_recognizer:
                self.keyword_recognizer.push_audio(audio_data)
        except Exception as e:
            logger.error(f"Error processing VAD: {e}")
            speech_detected, is_speech = False, False

        if speech_detected or self.speech_started:
            if is_speech:
                if not self.speech_started:
                    logger.info("Speech started")
                    self.buffer_pointer = self._update_buffer(
                        audio_data,
                        self.audio_buffer,
                        self.buffer_pointer,
                        self.buffer_size,
                    )
                    current_buffer = self._get_buffer_content(
                        self.audio_buffer, self.buffer_pointer, self.buffer_size
                    ).copy()

                    fade_length = min(
                        self.cross_fade_samples, len(current_buffer), len(audio_data)
                    )
                    if fade_length > 0:
                        fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
                        fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)

                        buffer_fade_section = current_buffer[-fade_length:].astype(
                            np.float32
                        )
                        audio_fade_section = audio_data[:fade_length].astype(np.float32)

                        faded_buffer_section = buffer_fade_section * fade_out
                        faded_audio_section = audio_fade_section * fade_in

                        current_buffer[-fade_length:] = np.round(
                            faded_buffer_section
                        ).astype(np.int16)
                        audio_data[:fade_length] = np.round(faded_audio_section).astype(
                            np.int16
                        )

                        combined_audio = np.concatenate((current_buffer, audio_data))
                    else:
                        combined_audio = audio_data

                    logger.info("Sending buffered audio to client via event handler...")
                    self.event_handler.on_speech_start()
                    self.event_handler.send_audio_data(combined_audio.tobytes())
                    if self.enable_wave_capture and self.wave_file:
                        try:
                            self.wave_file.writeframes(combined_audio.tobytes())
                        except Exception as e:
                            logger.error(f"Error writing to wave file: {e}")
                else:
                    self.event_handler.send_audio_data(audio_data.tobytes())
                    if self.enable_wave_capture and self.wave_file:
                        try:
                            self.wave_file.writeframes(audio_data.tobytes())
                        except Exception as e:
                            logger.error(f"Error writing to wave file: {e}")
                self.speech_started = True
            else:
                logger.info("Speech ended")
                self.event_handler.on_speech_end()
                self.speech_started = False

        if self.vad:
            self.buffer_pointer = self._update_buffer(
                audio_data, self.audio_buffer, self.buffer_pointer, self.buffer_size
            )

        return (None, pyaudio.paContinue)

    def _update_buffer(
        self, new_audio: np.ndarray, buffer: np.ndarray, pointer: int, buffer_size: int
    ) -> int:
        """
        Updates the internal audio buffer with new audio data.

        :param new_audio: New incoming audio data as a NumPy array.
        :param buffer: Internal circular buffer as a NumPy array.
        :param pointer: Current pointer in the buffer.
        :param buffer_size: Total size of the buffer.
        :return: Updated buffer pointer.
        """
        new_length = len(new_audio)
        if new_length >= buffer_size:
            buffer[:] = new_audio[-buffer_size:]
            pointer = 0
            logger.debug("Buffer overwritten with new audio data.")
        else:
            end_space = buffer_size - pointer
            if new_length <= end_space:
                buffer[pointer : pointer + new_length] = new_audio
                pointer += new_length
                logger.debug(f"Buffer updated. New pointer position: {pointer}")
            else:
                buffer[pointer:] = new_audio[:end_space]
                remaining = new_length - end_space
                buffer[:remaining] = new_audio[end_space:]
                pointer = remaining
                logger.debug(f"Buffer wrapped around. New pointer position: {pointer}")
        return pointer

    def _get_buffer_content(
        self, buffer: np.ndarray, pointer: int, buffer_size: int
    ) -> np.ndarray:
        """
        Retrieves the current content of the buffer in the correct order.

        :param buffer: Internal circular buffer as a NumPy array.
        :param pointer: Current pointer in the buffer.
        :param buffer_size: Total size of the buffer.
        :return: Ordered audio data as a NumPy array.
        """
        if pointer == 0:
            logger.debug("Buffer content retrieved without wrapping.")
            return buffer.copy()
        logger.debug("Buffer content retrieved with wrapping.")
        return np.concatenate((buffer[pointer:], buffer[:pointer]))

    def _on_keyword_detected(self, result):
        """
        Internal callback when a keyword is detected.
        """
        logger.info("Keyword detected")
        if self.keyword_recognizer:
            try:
                self.keyword_recognizer.stop_recognition()
                self.event_handler.on_keyword_detected(result)
                self.keyword_recognizer.start_recognition()
                logger.debug("Keyword recognizer restarted after detection.")
            except Exception as e:
                logger.error(f"Error handling keyword detection: {e}")

    def close(self):
        """
        Closes the audio capture stream and the wave file, releasing all resources.
        """
        self.stop(terminate=True)
        logger.info("AudioCapture resources have been released.")

    def __del__(self):
        self.close()
