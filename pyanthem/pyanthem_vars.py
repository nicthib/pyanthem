from os.path import realpath, join, split
self_fns = {
'fr':'entry',
'start_percent':'entry',
'end_percent':'entry',
'baseline':'entry',
'brightness':'entry',
'threshold':'entry',
'octave_add':'entry',
'scale_type':'entry',
'key':'entry',
'audio_format':'entry',
'Wshow':'entry',
'cmapchoice':'entry', 
'speed':'entry',
'file_in':'entry',
'file_out':'entry',
'save_path':'entry',
'Wshow_arr':'value'
}

octave_add_opts = {
'0':0,
'1':12,
'2':24,
'3':36,
'4':48,
'5':60,
'6':60
}

scale_keys = {
'Chromatic (12/oct)':[0,1,2,3,4,5,6,7,8,9,10,11],
'Major scale (7/oct)':[0,2,4,5,7,9,11],
'Minor scale (7/oct)':[0,2,3,5,7,9,11],
'Maj. chord (3/oct)':[0,4,7],
'Min. chord (3/oct)':[0,3,7],
'Maj. 7 (4/oct)':[0,4,7,11],
'Min. 7 (4/oct)':[0,3,7,11],
'Maj. 2/9 (4/oct)':[0,2,4,7,11],
'Min. 2/9 (4/oct)':[0,2,3,7,11]
}

key_opts={
'C' :0, 
'C#':1,
'D' :2,
'D#':3,
'E' :4,
'F' :5,
'F#':6,
'G' :7,
'G#':8,
'A' :9,
'A#':10,
'B' :11
}

cmaps_opts = tuple(sorted(['viridis', 'plasma', 'inferno', 'magma', 'cividis','binary', 
'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool','hot','copper','Spectral', 
'coolwarm', 'bwr', 'seismic','twilight', 'hsv', 'Paired', 'prism', 'ocean', 
'terrain','brg', 'rainbow', 'jet'],key=lambda s: s.lower()))

pth=split(realpath(__file__))[0]
example_files_decomp = [join(pth,'anthem_datasets',d) for d in ['demo1.mat','demo2.mat','demo3.mat','demo4.mat']]
example_files_raw = [join(pth,'anthem_datasets',d) for d in ['raw1.mat','raw2.mat']]
example_cfg = [join(pth,'anthem_datasets',d) for d in ['demo1_cfg.p','demo2_cfg.p','demo3_cfg.p','demo4_cfg.p']]

'''
Google drive id's for .sf2 files from https://sites.google.com/site/soundfonts4u/
'''
sound_fonts={
'piano_small':'12WYF3pc_kYI5myMjgEUnb-y2CeX3fucx',
'e-piano':'0B4_6p-MMrzwLeTBfNzl1SmVVSEU',
'strings':'1c0pCI0YdcFEpSLEbCW8HTzFOlJpz0HS9'
}

'''
static variables
'''

C0 = 16.352
fs = 44100
