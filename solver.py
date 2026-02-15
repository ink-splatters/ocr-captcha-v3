from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from mps import to_mps


class Solver:
    def __init__(
        self, model: VisionEncoderDecoderModel, processor: TrOCRProcessor, use_mps: bool = True
    ):

        self._model = to_mps(model) if use_mps else model
        self._processor = processor

    def solve(self, image: Image.Image) -> str:
        # Prepare image
        pixel_values = to_mps(self._processor(image, return_tensors="pt").pixel_values)

        # Generate text
        generated_ids = self._model.generate(pixel_values)
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
