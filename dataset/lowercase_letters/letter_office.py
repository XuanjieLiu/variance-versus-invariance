import argparse
import os
import string
from random import shuffle

from tqdm import tqdm

from dataset.phonenums.typography import Typography


def get_fitting_font_size(font, alphabet, patch_width, patch_height, margin=2):
    from PIL import ImageFont

    max_width = patch_width - 2 * margin
    max_height = patch_height - 2 * margin
    for font_size in range(patch_height, 1, -1):
        fnt = ImageFont.truetype(font, font_size)
        fits = True
        for char in alphabet:
            left, top, right, bottom = fnt.getbbox(char)
            if right - left > max_width or bottom - top > max_height:
                fits = False
                break
        if fits:
            return font_size
    return 1


class LetterOffice:
    def __init__(
        self,
        pagesize="26x1",
        patchsize="32x48",
        font="./dataset/phonenums/fonts/ITCKRIST.TTF",
        output_dir=None,
        alphabet=string.ascii_lowercase,
        dataset_name="lowercase letter",
        margin=2,
    ) -> None:
        self.typography = Typography(pagesize=pagesize, patchsize=patchsize)
        self.font = font
        self.output_dir = output_dir
        self.alphabet = list(alphabet)
        self.dataset_name = dataset_name
        self.margin = margin
        self.font_size = get_fitting_font_size(
            font,
            self.alphabet,
            self.typography.patch_width,
            self.typography.patch_height,
            margin=margin,
        )

    def generate_folder(self, n_pages=26000):
        """
        Generate images with all target letters once per sample.
        The color is the style label, matching the PhoneNums setup.
        """
        from matplotlib import colors

        os.makedirs(self.output_dir, exist_ok=True)
        palette = [
            "black",
            "blue",
            "green",
            "red",
            "teal",
            "purple",
            "orange",
            "brown",
        ]

        for i in tqdm(range(n_pages)):
            letters = self.alphabet.copy()
            shuffle(letters)
            text = "".join(letters)

            style = palette[i % len(palette)]
            color = list(colors.to_rgb(style))
            color = tuple([int(c * 240 + 8) for c in color])
            output_name = f"{text}_{style}.png"
            output_path = os.path.join(self.output_dir, output_name)
            self.typography.printer(
                text,
                font=self.font,
                fg=color,
                output_path=output_path,
                font_size=self.font_size,
                fit_to_bbox=True,
                margin=self.margin,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lowercase letter images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/lowercase_letters_",
        help="Directory to save generated lowercase letter images",
    )
    parser.add_argument(
        "--n_pages",
        type=int,
        default=26000,
        help="Number of pages to generate",
    )
    args = parser.parse_args()

    press = LetterOffice(
        pagesize="26x1",
        patchsize="32x48",
        output_dir=args.save_dir,
        alphabet=string.ascii_lowercase,
        dataset_name="lowercase letter",
    )
    press.generate_folder(n_pages=args.n_pages)
    print(f"Generated {args.n_pages} pages of lowercase letter images")
    print(f"Saved to {args.save_dir}")
