"""Legacy VERL dataset extension with explicit, consumed image-area controls.

VERL 0.7's standard process_vision_info ignores arbitrary data.image_kwargs.
This class places min/max pixel settings on each actual image content item before
the official qwen-vl-utils decoder. No frame is removed or reordered.
"""
import copy

from verl.utils.dataset.rl_dataset import RLHFDataset


class BoundedSpatialDataset(RLHFDataset):
    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        from qwen_vl_utils import process_vision_info
        limits = config.get('image_kwargs', {})
        if set(limits) != {'min_pixels', 'max_pixels'}:
            raise ValueError('Explicit min_pixels/max_pixels required')
        if not 0 < int(limits['min_pixels']) <= int(limits['max_pixels']):
            raise ValueError('Invalid image pixel limits')
        bounded = copy.deepcopy(messages)
        image_count = 0
        for message in bounded:
            if isinstance(message.get('content'), list):
                for item in message['content']:
                    if item.get('type') == 'image':
                        item.update({key: int(value) for key, value in limits.items()})
                        image_count += 1
        images, videos = process_vision_info(bounded, image_patch_size=image_patch_size, return_video_metadata=True)
        if len(images or []) != image_count:
            raise ValueError('Image decoding changed frame count')
        return images, videos
