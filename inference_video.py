import argparse
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
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
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
    parser.add_argument('--outscale', type=float, help='The final upsampling scale of the image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--vcodec', default='hevc_nvenc', help='Output video codec')
    parser.add_argument('--pix_fmt', default='yuv420p', help='Output video chroma subsampling')
    parser.add_argument('--amp', dest='amp', help='Enable automatic mixed precision inferencing', action='store_true')
    args = parser.parse_args()

    # model params
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
        print(f"Unknown model: {args.model_path}")
        sys.exit(2)

    if args.amp:
        autocast = torch.cuda.amp.autocast()
    else:
        autocast = contextlib.nullcontext()

    outscale = outscale if args.outscale else netscale

    # Turn on cuDNN benchmark and optimization.
    # ESRGAN input size is fixed, so cuDNN benefits subsequent executions.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        dni_weight=None,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=0,
        half=args.amp)

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=netscale,
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
    frame_pix_fmt = 'bgr24'  # Real-ESRGAN uses bgr24 input/output

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

    # Calculate ffmpeg video filter (e.g. resize, padding, etc.)
    out_filter_chain=[]
    out_width, out_height = int(outscale * width), int(outscale  * height)
    if netscale != outscale:
        # outscale is different from netscale
        print(f"FYI: outscale={outscale} doesn't match netscale={netscale}, ffmpeg will scale the output.")
        out_filter_chain.append(f'scale={out_width}x{out_height}:flags=lanczos')
    if out_width % 2 != 0 or out_height % 2 != 0:
        # output needs padding to fit meet encoder's expectation
        out_filter_chain.append(f'pad=ceil(iw/2)*2:ceil(ih/2)*2')

    # Pick codec params
    codec_args = []
    if args.vcodec == 'hevc_nvenc':
        codec_args = ['-c:v', args.vcodec, '-preset', 'p7', '-cq', '18']
    elif args.vcodec == 'libx265':
        codec_args = ['-c:v', args.vcodec, '-crf', '18']
    else:
        print("FYI: unknown vcodec={vcodec}, default ffmpeg codec params will be used")
        codec_args = ['-c:v', args.vcodec]

    # Spawn a background ffmpeg to encode output image to output video.
    net_width, net_height = int(netscale * width), int(netscale * height)
    net_frame_byte_size = net_width * net_height * n_channels
    output_proc = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-nostats',
            '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-r', fps, '-video_size', f'{net_width}x{net_height}',
            '-i', '-',
        ]
        + ([] if not has_audio_stream else ['-i', args.input, '-map', '0:v', '-map', '1:a', '-c:a', 'copy'])
        + ([] if not out_filter_chain else ['-vf', ','.join(out_filter_chain)])
        + codec_args + ['-pix_fmt', args.pix_fmt]
        + [
            '-y', args.output
        ],
        stdin = subprocess.PIPE,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        encoding = None,
        bufsize = 2 * net_frame_byte_size
    )

    print(f'Streamers started, input pid: {input_proc.pid}, output pid: {output_proc.pid}')

    # Include a GPU saturation measurement in stat.
    pbar = tqdm(total=frames, ascii=True)

    # Performance metrics to evaluate GPU saturation.
    perf_read_start = time.time()

    while buf := input_proc.stdout.read(frame_byte_size):
        raw_frame = np.reshape(
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
                    output, _ = upsampler.enhance(raw_frame, outscale=None)
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
