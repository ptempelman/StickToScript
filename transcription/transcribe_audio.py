import os.path as osp

import whisper
import content_generation as content_dir_parent


def transcribe_audio():
    model = whisper.load_model("medium")

    result = model.transcribe(
        audio=osp.join(
            osp.dirname(content_dir_parent.__file__),
            ".content",
            "full_conversation.mp3",
        ),
        fp16=False,
    )
    return result["text"]
