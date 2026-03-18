import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from ..util.wrapper_base import copy_obj_and_mtl, get_textures_from_obj

from .realesrgan import RealESRGANer


def fix_relative_path(path):
    # pathlibで統一して簡潔にする
    return str(Path(path).expanduser().resolve())


def sorted_glob(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(files)


def main():
    """Inference demo for Real-ESRGAN."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="input", help="Input image or folder"
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="RealESRGAN_x4plus",
        help=(
            "Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | "
            "realesr-animevideov3 | realesr-general-x4v3"
        ),
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output folder"
    )
    parser.add_argument(
        "-dn",
        "--denoise_strength",
        type=float,
        default=0.5,
        help=(
            "Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. "
            "Only used for the realesr-general-x4v3 model"
        ),
    )
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=4,
        help="The final upsampling scale of the image",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Option] Model path. Usually, you do not need to specify it",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix of the restored image"
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=0,
        help="Tile size, 0 for no tile during testing",
    )
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    parser.add_argument(
        "--pre_pad", type=int, default=0, help="Pre padding size at each border"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 precision during inference. Default: fp16 (half precision).",
    )
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="The upsampler for the alpha channels. Options: realesrgan | bicubic",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=None,
        help="gpu device to use (default=None) can be 0,1,2 for multi-gpu",
    )
    parser.add_argument(
        "--input-ext", type=str, default="png", help="Input image extension"
    )

    args = parser.parse_args()
    args.input = fix_relative_path(args.input)
    args.output = fix_relative_path(args.output)

    # determine models according to model names
    args.model_name = args.model_name.split(".")[0]
    netscale = None
    file_url = []
    if args.model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif args.model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif (
        args.model_name == "RealESRGAN_x4plus_anime_6B"
    ):  # x4 RRDBNet model with 6 blocks
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif args.model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif args.model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif args.model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join("weights", args.model_name + ".pth")
        if not os.path.isfile(model_path):
            # RealESRGANer will handle the download if we pass the URL
            # But let's keep the logic here for multiple URLs (DNI)
            if len(file_url) > 0:
                model_path = file_url[0] if len(file_url) == 1 else file_url
            else:
                model_path = os.path.join("weights", args.model_name + ".pth")

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == "realesr-general-x4v3" and args.denoise_strength != 1:
        if isinstance(model_path, list):
            # already have both URLs
            pass
        else:
            wdn_model_path = model_path.replace(
                "realesr-general-x4v3", "realesr-general-wdn-x4v3"
            )
            model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )

    os.makedirs(args.output, exist_ok=True)

    if args.ext == "auto":
        extension = args.input_ext
    else:
        extension = args.ext

    paths = []
    # 1. OBJ からテクスチャを特定する
    input_path = Path(args.input)
    if input_path.is_dir():
        for obj_file in input_path.glob("*.obj"):
            for tex_path in get_textures_from_obj(obj_file):
                if tex_path.exists():
                    paths.append(str(tex_path))
    elif input_path.suffix.lower() == ".obj":
        for tex_path in get_textures_from_obj(input_path):
            if tex_path.exists():
                paths.append(str(tex_path))

    paths = sorted(list(set(paths)))

    isatty = sys.stdout.isatty()
    pbar = tqdm(
        total=len(paths),
        unit="file",
        leave=False,
        dynamic_ncols=isatty,
        disable=not isatty,
    )
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        pbar.set_description(imgname)
        if not isatty:
            print(f"Processing {imgname}")

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            output = img
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
            output = img
        finally:
            if args.ext == "auto":
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == "RGBA":  # RGBA images should be saved in png format
                extension = "png"

            # 新仕様: appearance 直下に bldg_id.ext で保存
            save_path = os.path.join(
                args.output, "appearance", f"{imgname}.{extension}"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, output)

            pbar.update(1)
    pbar.close()

    # OBJ と MTL をコピー
    copy_obj_and_mtl(args.input, args.output, extension)


if __name__ == "__main__":
    main()
