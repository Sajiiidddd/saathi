"""Live captions, pushed to the browser over the WebRTC data channel.

Polling the store can never feel live — interim transcripts exist in the
pipeline the moment the caller starts a word, and TTS text arrives word by
word as she speaks. Two instances of this processor publish those events as
urgent transport messages (urgent = they bypass the frame queue, so a caption
never waits behind audio):

    stt  → CaptionPublisher(role suggested by frames: user interims/finals)
    tts  → CaptionPublisher(bot words + start/stop markers)

Message shape on the wire (JSON over the data channel):
    {"type": "caption", "role": "user", "text": "...", "final": false}
    {"type": "caption", "role": "bot",  "word": "...", "final": false}
    {"type": "caption", "role": "bot",  "reset": true}          # she starts
    {"type": "caption", "role": "user"|"bot", "final": true}

The client owns presentation (word fade-ins, solidifying interims); this side
only reports what is being said, as it is said.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class CaptionPublisher(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        message = None
        if isinstance(frame, InterimTranscriptionFrame):
            message = {"type": "caption", "role": "user", "text": frame.text or "", "final": False}
        elif isinstance(frame, TranscriptionFrame):
            message = {"type": "caption", "role": "user", "text": frame.text or "", "final": True}
        elif isinstance(frame, TTSStartedFrame):
            message = {"type": "caption", "role": "bot", "reset": True}
        elif isinstance(frame, TTSTextFrame):
            if frame.text:
                message = {"type": "caption", "role": "bot", "word": frame.text, "final": False}
        elif isinstance(frame, TTSStoppedFrame):
            message = {"type": "caption", "role": "bot", "final": True}
        if message is not None:
            await self.push_frame(OutputTransportMessageUrgentFrame(message=message))
