import numpy as np
from PIL import Image,ImageDraw
from spatial_intelligence.marker_cache import materialize


def test_lazy_renderer_matches_eager_pixels_and_order(tmp_path):
    paths=[]
    for index in range(4):
        path=tmp_path/f'{index}.jpg';Image.new('RGB',(64,64),(index*30,20,50)).save(path);paths.append(str(path))
    def render(images,info):
        ImageDraw.Draw(images[1]).rectangle((2,3,20,30),fill='red')
        array=np.array(images[3]);array[1:4]=255;images[3]=Image.fromarray(array)
    eager=[Image.open(p).convert('RGB') for p in paths];render(eager,{})
    output=materialize(paths,{},render,tmp_path/'cache')
    assert output[0]==paths[0] and output[2]==paths[2]
    for expected,actual in zip(eager,output):
        assert np.array_equal(np.array(expected),np.array(Image.open(actual)))
    assert materialize(paths,{},render,tmp_path/'cache')==output


def test_noop_multiframe_does_not_read_or_copy_images(tmp_path):
    paths=['not-opened-one','not-opened-two']
    assert materialize(paths,{},lambda images,info:None,tmp_path/'cache')==paths
