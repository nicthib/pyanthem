from os.path import realpath, join, split
self_fns={
'fr':'entry',
'start_percent':'entry',
'end_percent':'entry',
'baseline':'entry',
'brightness':'entry',
'threshold':'entry',
'octave_add':'entry',
'scale_type':'entry',
'key':'entry',
'sound_preset':'entry',
'comps_to_show':'entry',
'cmapchoice':'entry', 
'speed':'entry',
'file_in':'entry',
'file_out':'entry',
'save_path':'entry',
'audio_analog':'entry',
'comps_to_show_arr':'value',
'fluidsynthextracommand':'value'
}

octave_add_opts={
'0':0,
'1':12,
'2':24,
'3':36,
'4':48,
'5':60,
'6':60
}

scale_keys={
'Chromatic (12/oct)':[0,1,2,3,4,5,6,7,8,9,10,11],
'Major scale (7/oct)':[0,2,4,5,7,9,11],
'Minor scale (7/oct)':[0,2,3,5,7,9,11],
'Maj. chord (3/oct)':[0,4,7],
'Min. chord (3/oct)':[0,3,7],
'Maj. 7 (4/oct)':[0,4,7,11],
'Min. 7 (4/oct)':[0,3,7,11],
'Maj. 2/9 (5/oct)':[0,2,4,7,11],
'Min. 2/9 (5/oct)':[0,2,3,7,11]
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

cmaps_opts=tuple(sorted(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool','hot','copper', 
'coolwarm', 'bwr', 'seismic','twilight', 'hsv', 'prism', 'ocean', 
'terrain','brg', 'rainbow', 'jet'],key=lambda s: s.lower()))

'''
Google drive id for .sf2 files from https://sites.google.com/site/soundfonts4u/
'''
sound_font='1TvjLlgKqZxcrZvefciJGX1Q_3e8Cy4dP'
sound_presets={
'0':0,
'1':1,
'2':2,
'3':3,
'4':4,
'5':5,
'6':6,
'7':7,
'8':8,
'9':9,
'10':10,
}

'''
static variables
'''
fs=44100
