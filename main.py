import argparse
from pathlib import Path

import image as im
import solver as s
import pretrained as pt

DEFAULT_MODEL = "anuashok/ocr-captcha-v3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--vision-codec-model", dest="vc", default=DEFAULT_MODEL)
    p.add_argument("--tr-ocrp-model", dest="tr", default=DEFAULT_MODEL)

    args = p.parse_args()
    image = im.load(args.image)

    ps = pt.load_from_hf(vision_codec=args.vc, tr_ocrp=args.tr)

    solver = s.Solver(ps)

    print(solver.solve(image))


if __name__ == "__main__":
    main()
