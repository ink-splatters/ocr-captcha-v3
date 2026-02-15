# ocr-captcha-v3

python project for [ocr-captcha-v3](https://huggingface.co/anuashok/ocr-captcha-v3) inferrence.

## Setup

```sh
uv sync
uv run poe download # not necessary, to cache model before running the inferrence
```

## HW acceleration

Currently only MPS is supported (tried by default).

## Usage

```sh
uv run main.py <captcha image file>
```

## Dev

```sh
uv sync --dev
uv run poe fix # runs `ruff` and `ty` on the project
```

## License

MIT
