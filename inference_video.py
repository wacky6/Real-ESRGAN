import argparse
import cv2
import os
import subprocess
import sys
import json
import numpy as np
from tqdm import tqdm
import time
import torch
import contextlib

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input video')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument('--block', type=int, default=23, help='num_block in RRDB')
    parser.add_argument('--vcodec', default='libx264', help='Output video codec')
    parser.add_argument('--vpreset', default='veryslow', help='Output video preset')
    parser.add_argument('--vcrf', default='19', help='Output video CRF')
    parser.add_argument('--vpix_fmt', default='yuv420p', help='Output video chroma subsampling')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument('--amp', dest='amp', help='Enable automatic mixed precision inferencing', action='store_true')
    args = parser.parse_args()

    if 'RealESRGAN_x4plus_anime_6B.pth' in args.model_path:
        args.block = 6
    elif 'RealESRGAN_x2plus.pth' in args.model_path:
        args.netscale = 2

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32, scale=args.netscale)

    if args.amp:
        autocast = torch.cuda.amp.autocast()
    else:
        autocast = contextlib.nullcontext()

    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # Probe video stream.
    probe_proc = subprocess.run([
            'ffprobe',
            '-hide_banner', '-select_streams', 'v:0', '-show_streams',
            '-print_format', 'json',
            args.input
        ],
        capture_output=True,
        encoding = 'utf-8'
    )

    if probe_proc.returncode != 0:
        print('Fail to probe video file, raw error: ', file=sys.stderr)
        print(probe_proc.stderr, file=sys.stderr)
        exit(probe_proc.returncode)

    video_streams = json.loads(probe_proc.stdout)
    if len(video_streams["streams"]) != 1:
        print("Can't find video stream, probed: ", file=sys.stderr)
        print(probe_proc.stdout)
        exit(2)

    # Same as ffprobe's json output.
    video_stream = video_streams['streams'][0]

    width, height = video_stream['width'], video_stream['height'],
    # TODO: nb_frames is known to be inaccurate, perhaps also probe based on frame_rate * duration
    frames = int(video_stream['nb_frames']) if 'nb_frames' in video_stream else None
    fps = video_stream['r_frame_rate']
    n_channels = 3  # RGB channels


    print(f'Found video stream: {width}x{height} size, {frames if frames else "<unknown>"} frames @ {fps} fps')

    # Probe audio stream. If valid, automatically add audio to output.
    probe_audio_proc = subprocess.run([
            'ffprobe',
            '-hide_banner', '-select_streams', 'a:0', '-show_streams',
            '-print_format', 'json',
            args.input
        ],
        capture_output=True,
        encoding = 'utf-8'
    )
    audio_streams = json.loads(probe_audio_proc.stdout)
    has_audio_stream = len(audio_streams["streams"]) > 0

    # Spawn a background ffmpeg to read video stream to rawvideo frames.
    frame_byte_size = width * height * n_channels
    frame_pix_fmt = 'bgr24'  # OpenCV uses BGR ordering

    ### TODO: Add SIGINT / SIGTERM handler to kill input/output proc

    input_proc = subprocess.Popen([
            'ffmpeg', '-hide_banner',
            '-i', args.input, '-map', '0:v:0',
            '-c:v', 'rawvideo', '-pix_fmt', frame_pix_fmt,
            '-f', 'image2pipe', '-',
        ],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        encoding = None,
        # Buffer a few rawvideo frrames
        bufsize = 4 * frame_byte_size,
    )

    # Spawn a background ffmpeg to encode output image to output video.
    out_width, out_height = int(args.outscale * width), int(args.outscale * height)
    out_frame_byte_size = out_width * out_height * n_channels
    # TODO: make padding intelligent based on common video dimensions?
    need_padding = out_width % 2 != 0 or out_height % 2 != 0
    output_proc = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-nostats',
            '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-r', fps, '-video_size', f'{out_width}x{out_height}',
            '-i', '-',
        ]
        + ([] if not has_audio_stream else ['-i', args.input, '-map', '0:v', '-map', '1:a', '-c:a', 'copy'])
        + ([] if not need_padding else ['-vf', f'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        + [
            '-c:v', args.vcodec, '-crf', args.vcrf, '-preset', args.vpreset, '-pix_fmt', args.vpix_fmt,
            '-y', args.output
        ],
        stdin = subprocess.PIPE,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        encoding = None,
        bufsize = 2 * out_frame_byte_size
    )

    print(f'Streamers started, input pid: {input_proc.pid}, output pid: {output_proc.pid}')

    # Include a GPU saturation measurement in stat.
    pbar = tqdm(total=frames, ascii=True)

    # Performance metrics to evaluate GPU saturation.
    perf_read_start = time.time()

    while buf := input_proc.stdout.read(frame_byte_size):
        raw_frame = np.resize(
            np.frombuffer(buf, dtype=np.uint8),
            (height, width, n_channels)
        )

        perf_read_time = time.time() - perf_read_start
        perf_proc_start = time.time()

        # Upscale.
        try:
            with autocast as autocast_:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(
                        raw_frame,
                        has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    output, _ = upsampler.enhance(raw_frame, outscale=args.outscale)
        except Exception as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            input_proc.kill()
            output_proc.kill()
            exit(5)

        # Check if output streamer is dead, if so terminate early.
        if output_proc.poll():
            pbar.close()
            print(f'Output streamer is dead, exit code: {output_proc.returncode}')
            input_proc.kill()
            exit(5)

        perf_proc_time = time.time() - perf_proc_start
        perf_write_start = time.time()

        output_proc.stdin.write(output.tobytes())

        perf_write_time = time.time() - perf_write_start
        pbar.set_postfix({
            'p_util': f'{perf_proc_time / (perf_read_time + perf_write_time + perf_proc_time) * 100:.1f}%',
        })
        pbar.update()

        perf_read_start = time.time()

    # Mark the end of frames.
    output_proc.stdin.close()

    # Wait for streamers to exit.
    input_proc.wait()
    output_proc.wait()
    pbar.close()
    print(f'Streamers completed, input exit code: {input_proc.returncode}, output exit code: {output_proc.returncode}')

if __name__ == '__main__':
    main()
