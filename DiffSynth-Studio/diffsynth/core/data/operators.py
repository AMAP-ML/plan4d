import math
import torch, torchvision, imageio, os
import imageio.v3 as iio
import numpy as np
from PIL import Image
import torchaudio


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_rate=24, fix_frame_rate=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        duration = meta_data["duration"] if "duration" in meta_data else total_original_frames / meta_data['fps']
        total_available_frames = math.floor(duration * self.frame_rate)
        return int(total_available_frames)

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))        
        frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        num_frames = self.get_num_frames(reader)
        total_raw_frames = reader.count_frames()
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class LoadPointOdysseyClip(DataProcessingOperator):
    """
    Online Point Odyssey clip loader driven by metadata row:
    - scene_dir
    - start_idx
    - end_idx

    This operator can output one field among:
    rgbs / depth / ray / intrinsics / extrinsics / valid_mask
    and is intended to be used through UnifiedDataset special_operator_map.
    """

    def __init__(
        self,
        dataset_base_path="",
        scene_dir_key="scene_dir",
        start_idx_key="start_idx",
        end_idx_key="end_idx",
        target_height=None,
        target_width=None,
    ):
        self.dataset_base_path = dataset_base_path
        self.scene_dir_key = scene_dir_key
        self.start_idx_key = start_idx_key
        self.end_idx_key = end_idx_key
        self.target_height = target_height
        self.target_width = target_width
        self._cache_signature = None
        self._cache_outputs = None
        try:
            import cv2 as _cv2
        except ImportError as error:
            raise ImportError(
                "LoadPointOdysseyClip requires OpenCV (`cv2`) for image/depth loading."
            ) from error
        self._cv2 = _cv2

    @staticmethod
    def _compute_resize_params(src_height, src_width, target_height, target_width):
        scale_h = target_height / src_height
        scale_w = target_width / src_width
        scale = max(scale_h, scale_w)
        resize_height = int(np.ceil(src_height * scale))
        resize_width = int(np.ceil(src_width * scale))
        resize_height = max(resize_height, target_height)
        resize_width = max(resize_width, target_width)
        crop_top = (resize_height - target_height) // 2
        crop_left = (resize_width - target_width) // 2
        return resize_height, resize_width, crop_top, crop_left, target_height, target_width

    def _resize_and_center_crop_image(self, image, target_height, target_width, interpolation):
        src_height, src_width = image.shape[:2]
        params = LoadPointOdysseyClip._compute_resize_params(
            src_height, src_width, target_height, target_width
        )
        resize_height, resize_width, crop_top, crop_left, crop_h, crop_w = params
        resized = self._cv2.resize(image, (resize_width, resize_height), interpolation=interpolation)
        cropped = resized[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
        return cropped, params

    @staticmethod
    def _adjust_intrinsics_for_resize_and_crop(
        intrinsics,
        src_height,
        src_width,
        resize_height,
        resize_width,
        crop_top,
        crop_left,
    ):
        adjusted = intrinsics.copy()
        scale_x = resize_width / src_width
        scale_y = resize_height / src_height
        adjusted[0, 0] *= scale_x
        adjusted[1, 1] *= scale_y
        adjusted[0, 2] *= scale_x
        adjusted[1, 2] *= scale_y
        adjusted[0, 2] -= crop_left
        adjusted[1, 2] -= crop_top
        return adjusted

    @staticmethod
    def _generate_ray_map(intrinsics, extrinsics, height, width):
        device = intrinsics.device
        dtype = intrinsics.dtype
        u = torch.arange(width, dtype=dtype, device=device)
        v = torch.arange(height, dtype=dtype, device=device)
        u, v = torch.meshgrid(u, v, indexing="xy")
        ones = torch.ones_like(u)
        pixels = torch.stack([u, v, ones], dim=-1)

        k_inv = torch.inverse(intrinsics)
        directions_cam = torch.einsum("ij,hwj->hwi", k_inv, pixels)
        r = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        r_inv = r.T
        directions_world = torch.einsum("ij,hwj->hwi", r_inv, directions_cam)
        origin = -r_inv @ t
        origins = origin.view(1, 1, 3).expand(height, width, 3)

        ray_map = torch.cat([directions_world, origins], dim=-1)
        return ray_map.permute(2, 0, 1)

    def _resolve_scene_dir(self, sample):
        if self.scene_dir_key not in sample:
            raise KeyError(f"Missing `{self.scene_dir_key}` in metadata row.")
        raw_scene_dir = str(sample[self.scene_dir_key])
        if os.path.isabs(raw_scene_dir):
            return raw_scene_dir
        return os.path.abspath(os.path.join(self.dataset_base_path, raw_scene_dir))

    @staticmethod
    def _load_rgb_path(scene_dir, frame_idx):
        jpg_path = os.path.join(scene_dir, "rgbs", f"rgb_{frame_idx:05d}.jpg")
        if os.path.exists(jpg_path):
            return jpg_path
        png_path = os.path.join(scene_dir, "rgbs", f"rgb_{frame_idx:05d}.png")
        if os.path.exists(png_path):
            return png_path
        raise FileNotFoundError(f"RGB frame not found: frame={frame_idx} scene={scene_dir}")

    @staticmethod
    def _load_depth_path(scene_dir, frame_idx):
        depth_path = os.path.join(scene_dir, "depths", f"depth_{frame_idx:05d}.png")
        return depth_path if os.path.exists(depth_path) else None

    def _load_clip_from_sample(self, sample):
        if self.start_idx_key not in sample or self.end_idx_key not in sample:
            raise KeyError(
                f"Missing `{self.start_idx_key}` / `{self.end_idx_key}` in metadata row."
            )
        scene_dir = self._resolve_scene_dir(sample)
        start_idx = int(sample[self.start_idx_key])
        end_idx = int(sample[self.end_idx_key])
        if end_idx <= start_idx:
            raise ValueError(
                f"Invalid clip range: start_idx={start_idx}, end_idx={end_idx}"
            )

        signature = (scene_dir, start_idx, end_idx, self.target_height, self.target_width)
        if self._cache_signature == signature and self._cache_outputs is not None:
            return self._cache_outputs

        anno_path = os.path.join(scene_dir, "anno.npz")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"anno.npz not found: {anno_path}")

        use_resize_crop = self.target_height is not None and self.target_width is not None
        if (self.target_height is None) ^ (self.target_width is None):
            raise ValueError("target_height and target_width must be both set or both unset.")

        rgbs = []
        depths = []
        rays = []
        intrinsics_list = []
        extrinsics_list = []

        with np.load(anno_path) as anno:
            for frame_idx in range(start_idx, end_idx):
                rgb_path = self._load_rgb_path(scene_dir, frame_idx)
                rgb = self._cv2.imread(rgb_path)
                if rgb is None:
                    raise RuntimeError(f"Failed to read RGB frame: {rgb_path}")
                rgb = self._cv2.cvtColor(rgb, self._cv2.COLOR_BGR2RGB)

                orig_h, orig_w = rgb.shape[:2]
                intrinsics_orig = anno["intrinsics"][frame_idx].copy()
                extrinsics = anno["extrinsics"][frame_idx].copy()

                if use_resize_crop:
                    rgb, resize_params = self._resize_and_center_crop_image(
                        rgb,
                        self.target_height,
                        self.target_width,
                        interpolation=self._cv2.INTER_LINEAR,
                    )
                    resize_h, resize_w, crop_top, crop_left, _, _ = resize_params
                    intrinsics = self._adjust_intrinsics_for_resize_and_crop(
                        intrinsics_orig,
                        orig_h,
                        orig_w,
                        resize_h,
                        resize_w,
                        crop_top,
                        crop_left,
                    )
                else:
                    intrinsics = intrinsics_orig

                depth_path = self._load_depth_path(scene_dir, frame_idx)
                if depth_path is not None:
                    depth = self._cv2.imread(depth_path, self._cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        raise RuntimeError(f"Failed to read depth frame: {depth_path}")
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) / 1000.0
                    else:
                        depth = depth.astype(np.float32)
                    if use_resize_crop:
                        depth, _ = self._resize_and_center_crop_image(
                            depth,
                            self.target_height,
                            self.target_width,
                            interpolation=self._cv2.INTER_NEAREST,
                        )
                else:
                    h, w = rgb.shape[:2]
                    depth = np.zeros((h, w), dtype=np.float32)

                h, w = rgb.shape[:2]
                ray = self._generate_ray_map(
                    intrinsics=torch.from_numpy(intrinsics).float(),
                    extrinsics=torch.from_numpy(extrinsics).float(),
                    height=h,
                    width=w,
                ).numpy()

                rgbs.append(rgb)
                depths.append(depth)
                rays.append(ray)
                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)

        rgbs_np = np.stack(rgbs, axis=0)
        depth_np = np.stack(depths, axis=0)
        ray_np = np.stack(rays, axis=0)
        intrinsics_np = np.stack(intrinsics_list, axis=0)
        extrinsics_np = np.stack(extrinsics_list, axis=0)
        valid_mask_np = (depth_np > 0).astype(np.uint8)

        outputs = {
            "rgbs": torch.from_numpy(rgbs_np),
            "depth": torch.from_numpy(depth_np).float(),
            "ray": torch.from_numpy(ray_np).float(),
            "intrinsics": torch.from_numpy(intrinsics_np).float(),
            "extrinsics": torch.from_numpy(extrinsics_np).float(),
            "valid_mask": torch.from_numpy(valid_mask_np),
        }

        self._cache_signature = signature
        self._cache_outputs = outputs
        return outputs

    def __call__(self, data, sample=None, key=None):
        if sample is None:
            raise ValueError(
                "LoadPointOdysseyClip requires `sample` context from UnifiedDataset."
            )
        outputs = self._load_clip_from_sample(sample)
        if key is None:
            return outputs
        if key not in outputs:
            raise KeyError(f"Unsupported PointOdyssey output key: {key}")
        return outputs[key]


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)

    def __call__(self, data: str):
        reader = self.get_reader(data)
        num_frames = self.get_num_frames(reader)
        duration = num_frames / self.frame_rate
        waveform, sample_rate = torchaudio.load(data)
        target_samples = int(duration * sample_rate)
        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, sample_rate
