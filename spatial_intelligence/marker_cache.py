"""Lazy, lossless materialization of only renderer-accessed image frames."""
import json
from pathlib import Path
from PIL import Image


class LazyImages:
    def __init__(self,paths):
        self.paths=list(paths);self.loaded={}

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        if not isinstance(index,int):raise TypeError('Renderer image index must be an integer')
        if index<0:index+=len(self.paths)
        if not 0<=index<len(self.paths):raise IndexError(index)
        if index not in self.loaded:
            with Image.open(self.paths[index]) as image:self.loaded[index]=image.convert('RGB')
        return self.loaded[index]

    def __setitem__(self,index,image):
        if not 0<=index<len(self.paths):raise IndexError(index)
        self.loaded[index]=image


def materialize(paths,info,renderer,folder):
    folder=Path(folder);marker=folder/'complete.json'
    if marker.exists():return json.loads(marker.read_text())['media']
    images=LazyImages(paths)
    renderer(images[0] if len(images)==1 else images,info)
    result=list(paths)
    folder.mkdir(parents=True,exist_ok=True)
    for index,image in images.loaded.items():
        target=folder/f'{index}.png';partial=folder/f'{index}.png.part'
        image.save(partial,format='PNG',compress_level=1);partial.replace(target)
        result[index]=str(target)
    partial=marker.with_suffix('.tmp')
    partial.write_text(json.dumps({'media':result,'renderer_cache':'lazy-lossless-v1',
                                   'materialized_frames':sorted(images.loaded)}))
    partial.replace(marker)
    return result
