import argparse
import os
import sys
import torch
import contextlib
import onnx

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', type=int, default=512, help="tile size")
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--netscale', type=int, default=4, help='Network scale, used for loading custom model')
    parser.add_argument('--amp', dest='amp', help='Enable automatic mixed precision inferencing', action='store_true')
    parser.add_argument('--exporter', choices=('torchscript', "dynamo"), default='torchscript')
    parser.add_argument('--output', type=str, default="/auto/")
    args = parser.parse_args()

    # Model params
    netscale = None
    if 'RealESRGAN_x4plus' in args.model_path:
        netscale = 4
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
    elif 'RealESRNet_x4plus' in args.model_path:
        netscale = 4
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
    elif 'RealESRGAN_x4plus_anime_6B' in args.model_path:
        netscale = 4
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=netscale)
    elif 'RealESRGAN_x2plus' in args.model_path:
        netscale = 2
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
    elif 'realesr-animevideov3' in args.model_path:
        netscale = 4
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    elif 'realesr-general-x4v3' in args.model_path:
        netscale = 4
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    elif 'realesr-general-wdn-x4v3' in args.model_path:
        netscale = 4
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    else:
        netscale = args.netscale
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)

    if args.amp:
        autocast = torch.cuda.amp.autocast()
    else:
        autocast = contextlib.nullcontext()

    # Trigger a model load.
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        dni_weight=None,
        model=model,
        tile=0,
        pre_pad=0,
        half=args.amp,
        device='cpu',
    )

    dtype = torch.float16 if args.amp else torch.float32
    dummy_input = torch.randn(1, 3, args.tile, args.tile, dtype=dtype)
    model = upsampler.model

    fp_spec = 16 if args.amp else 32
    input_names = [f'in_image_float{fp_spec}_rgb01']
    output_names = [f'out_image_float{fp_spec}_rbg01']

    if args.output == '/auto/':
        basename, _ = os.path.splitext(os.path.basename(args.model_path))
        args.output = f'{basename}_fp{fp_spec}_t{args.tile}_{args.exporter}.onnx'

    if args.exporter == 'torchscript':
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            verbose=True,
            opset_version=10,
            input_names=input_names,
            output_names=output_names,
        )
    elif args.exporter == 'dynamo':
        onnx_program = torch.onnx.dynamo_export(model, dummy_input)
        onnx_program.save(args.output)
    else:
        assert false, "not reached"

    # Check converted model is okay.
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)   # Throws on failure

    print("*** Conversion OK")

if __name__ == '__main__':
    main()