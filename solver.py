from PIL import Image

from device import to_best_available_device
from pretrained import PretrainedSet


class Solver:
    def __init__(self, ps: PretrainedSet):
        self._model = to_best_available_device(ps.model)
        self._processor = ps.processor

    def solve(self, image: Image.Image) -> str:
        # Prepare image
        pixel_values = self._processor(image, return_tensors="pt").pixel_values
        pixel_values = to_best_available_device(pixel_values)

        # Generate text
        generated_ids = self._model.generate(pixel_values)
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
