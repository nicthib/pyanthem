import pyanthem
from numpy.matlib import repmat
g=pyanthem.GUI(display=False)
import subprocess as sp
import numpy as np
file = r'C:\Users\dnt21\Downloads\stars.mp4'
command = [ 'ffmpeg','-i', file,'-f', 'image2pipe','-pix_fmt', 'rgb24','-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
vid = []
while True:
    try:
        raw_image = pipe.stdout.read(640*360*3)
        image =  np.fromstring(raw_image, dtype='uint8')
        vid.append(np.mean(image.reshape((360,640,3)),axis=2))
        pipe.stdout.flush()
    except:
        break
vid = np.moveaxis(np.asarray(vid),0,-1)
bl = np.mean(vid,axis=2)
vid = vid - np.repeat(bl[:, :, np.newaxis], vid.shape[2], axis=2)
g.process_raw(data=vid,frame_rate=25,save=True,n_clusters=30)