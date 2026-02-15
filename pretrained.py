import typing as t

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# OCR_CAPTCHA_V3 = "anuashok/ocr-captcha-v3"


class PretrainedSet(t.NamedTuple):
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor


def load_from_hf(
    *args: str, vision_codec: str | None = None, tr_ocrp: str | None = None
) -> PretrainedSet:
    if len(args) in (1, 2) and (vision_codec or tr_ocrp):
        raise ValueError("Cannot mix positional and keyword arguments")

    match len(args):
        case 0:
            if not (vision_codec and tr_ocrp):
                raise ValueError("Must provide both vision_codec and tr_ocrp")
        case 1:
            vision_codec = tr_ocrp = args[0]
        case 2:
            vision_codec, tr_ocrp = args
        case _:
            raise ValueError("Expected 0-2 positional args")

    return PretrainedSet(
        model=VisionEncoderDecoderModel.from_pretrained(vision_codec),  # ty: ignore[invalid-argument-type]
        processor=TrOCRProcessor.from_pretrained(tr_ocrp),  # ty: ignore[invalid-argument-type]
    )
