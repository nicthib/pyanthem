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
'Yamaha C5 Grand':0,
'Large Concert Grand':1,
'Mellow Grand':2,
'Bright C5 Grand':3,
'Upright Piano':4,
'Chateau Grand':5,
'Mellow Chateau Grand':6,
'Dark Chateau Grand':7,
'Rhodes EP':8,
'DX7 EP':9,
'Rhodes Bell EP':10,
'Rotary Organ':11,
'Small Pipe Organ':12,
'Pipe Organ Full':13,
'Small Plein-Jeu':14,
'Flute Sml Plein-Jeu':15,
'FlutePad Sml Plein-J':16,
'Plein-jeu Organ Lge':17,
'Pad Plein-Jeu Large':18,
'Warm Pad':19,
'Synth Strings':20,
'Voyager-8':21,
'Full Strings Vel':22,
'Full Orchestra':23,
'Chamber Strings 1':24,
'Chamber Str 2 (SSO)':25,
'Violin (all around)':26,
'Two Violins':27,
'Cello 1':28,
'Cello 2 (SSO)':29,
'Trumpet':30,
'Trumpet+8 Vel':31,
'Tuba':32,
'Oboe':33,
'Tenor Sax':34,
'Alto Sax':35,
'Flute Expr+8 (SSO)':36,
'Flute 2':37,
'Timpani':38,
'Banjo 5 String':39,
'Steel Guitar':40,
'Nylon Guitar':41,
'Spanish Guitar':42,
'Spanish V Slide':43,
'Clean Guitar':44,
'LP Twin Elec Gtr':45,
'LP Twin Dynamic':46,
'Muted LP Twin':47,
'Jazz Guitar':48,
'Chorus Guitar':49,
'YamC5 + Pad':50,
'YamC5+LowStrings':51,
'YamC5+ChamberStr':52,
'YamC5+Strings':53,
'Chateau Grand+Pad':54,
'Ch Grand+LowStrings':55,
'Ch Grand+ChamberStr':56,
'Ch Grand+Strings':57,
'DX7+Pad':58,
'DX7+LowStrings':59,
}

'''
static variables
'''
fs=44100
