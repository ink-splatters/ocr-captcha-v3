import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <captcha image>")
        SystemExit(1)

    import image as im

    image = im.load(Path(sys.argv[1]))

    import solver as s
    import pretrained as pt

    solver = s.Solver(pt.model, pt.processor)

    text: str = solver.solve(image)
    print(text)
