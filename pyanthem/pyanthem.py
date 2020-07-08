import os, random, sys, time, csv, pickle, re, pkg_resources
import mido
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']="hide"
from tkinter import StringVar, DoubleVar, Tk, Label, Entry, Button, OptionMenu, \
Checkbutton, Message, Menu, IntVar, Scale, HORIZONTAL, simpledialog, messagebox, Toplevel
from tkinter.ttk import Progressbar, Separator, Combobox
from tkinter import filedialog as fd 
import tkinter.font as font
from scipy.io import loadmat, savemat, whosmat
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from scipy.io.wavfile import write as wavwrite
from sklearn.cluster import KMeans
from pygame.mixer import init, quit, get_init, set_num_channels, pre_init, music
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as tkr
import matplotlib.cm as cmaps # https://matplotlib.org/gallery/color/colormap_reference.html
import numpy as np
from numpy.matlib import repmat
try:
	from pyanthem.pyanthem_vars import *
except:
	from pyanthem_vars import *
from google_drive_downloader import GoogleDriveDownloader as gdd
import subprocess as sp
import PIL.Image as Image

def init_entry(fn):
	'''
	Generalized version of StringVar/DoubleVar followed by set()
	'''
	if isinstance(fn, str) or fn is None:
		entry=StringVar()
	else:
		entry=DoubleVar()
	entry.set(fn)
	return entry

def stack_videos(videos,fn='output.mp4'):
	'''
	Stacks .mp4 videos horizontally (and combines audio)
	'''
	nvids=len(videos)
	instr=''
	for i in range(len(videos)):
		instr += ' -i '+videos[i]
	os.system('ffmpeg -y '+instr+' -filter_complex "[0:v][1:v]hstack=inputs='+str(nvids)+'[v]; [0:a][1:a]amerge[a]" -map "[v]" -map "[a]" -ac 2 '+fn)

def uiopen(title,filetypes):
	root=Tk()
	root.withdraw()
	file_in=os.path.normpath(fd.askopenfilename(title=title,filetypes=filetypes))
	root.update()
	root.destroy()
	return file_in

def run(display=True):
	'''
	Main command to run GUI or CLI
	'''
	root=GUI(display=display)
	sys.ps1='♫ '
	if display:
		root.mainloop()
	else:
		print('Welcome to pyanthem v{}!'.format(pkg_resources.require("pyanthem")[0].version))
		return root

class GUI(Tk):
	def __init__(self,display=True):
		'''
		Initializes the GUI instance. display=True runs the Tk.__init__(self)
		command, while display=False skips that and visual initialization, keeping
		the GUI 'hidden'
		'''
		self.display=display
		self.sf_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'anthem_soundfonts')
		if not os.path.isdir(self.sf_path):
			print('Initializing soundfont library...')
			os.mkdir(self.sf_path)
			gdd.download_file_from_google_drive(file_id=sound_font,dest_path=\
				os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'anthem_soundfonts'),'font.sf2'),showsize=True)
		if self.display:
			Tk.__init__(self)
			self.default_font=font.nametofont("TkDefaultFont")
			self.initGUI()
	
	def quit(self,event=None):
		'''
		Quits the GUI instance. Currently, jupyter instances are kinda buggy
		'''
		try:
			# This raises a NameError exception in a notebook env, since 
			# sys.exit() is not an appropriate method
			get_ipython().__class__.__name__ 
			self.destroy()
		except NameError:
			sys.exit()

	def message(self,msg):
		'''
		Sends message through print if no GUI, through self.status if GUI is running
		'''
		if self.display:
			self.status['text']='♫ '+msg
		else:
			print(msg)

	def check_data(self):
		'''
		Checks to make sure data is loaded.
		'''
		if hasattr(self,'data') and self.cfg['save_path'] is not None:
			return True
		else:
			self.message('Error: No dataset has been loaded or save_path is empty.')
			return False
	
	def self_to_cfg(self):
		'''
		This function is necessary to allow command-line access of the GUI functions. 
		StringVar() and IntVar() allow for dynamic, quick field updating and access, 
		but cannot be used outside of a mainloop or be pickled. for this reason, I convert 
		all StringVars and IntVars to a new dict called 'self.cfg', that can be accessed 
		oustide the GUI and dumped to a pickle file, which essentially "saves" the GUI.
		'''
		self.cfg={k: getattr(self,k).get() if self_fns[k] is 'entry' else getattr(self,k) for k in self_fns}

	def dump_cfg(self):
		'''
		Saves config file.
		'''
		file_out=os.path.join(self.cfg['save_path'],self.cfg['file_out'])+'_cfg.p'
		pickle.dump(self.cfg,open(file_out, "wb"))
		self.message(f'cfg file saved to {file_out}')
	
	def load_data(self,filein=None):
		'''
		loads dataset from filein. At the time, only supports .mat files.
		'''
		if filein is None:
			filein=uiopen(title='Select .mat file for import',filetypes=[('.mat files','*.mat')])
		if filein=='.':
			return
		data={}
		data_unk=loadmat(filein)
		for k in data_unk.keys():
			if k in ('__header__', '__version__', '__globals__'):
				continue
			elif len(data_unk[k].flatten())==1:
				data['fr']=float(data_unk[k])
			elif data_unk[k].ndim==2:
				data['H']=data_unk[k]
			elif data_unk[k].ndim==3:
				data['W']=data_unk[k]
		if ('H' in data and 'W' in data) and data['H'].shape[0] != data['W'].shape[-1] :
			self.message('Error: Inner dimensions of {} x {} do not match!')
			return
		if 'H' in data:
			if 'W' in data:
				data['W_shape']=data['W'].shape
				data['W']=data['W'].reshape(data['W'].shape[0]*data['W'].shape[1],data['W'].shape[2])
			if 'fr' not in data:
				data['fr']=data['H'].shape[1]/60 # Forces length to be 1 minute!
			self.data=data
			if not self.display:
				return self
		else:
			self.message('Error: .mat file incompatible. Please select a .mat file with three variables: W (3D), H (2D), and fr (1-element float)')

	def load_GUI(self):
		'''
		GUI-addons for load_data. Prompts user with filedialog, assigns defaults and sets GUI fields. 
		'''
		filein=uiopen(title='Select .mat file for import',filetypes=[('.mat files','*.mat')])
		if filein=='.':
			return
		self.load_data(filein)
		if not hasattr(self,'data'):
			return
		self.data['H_pp']=self.data['H']
		self.data['H_fp']=self.data['H']
		self.fr.set(self.data['fr'])
		if 'W' in self.data:
			self.data['W_pp']=self.data['W']
		self.file_in.set(os.path.splitext(os.path.split(filein)[1])[0])

		# Set some defaults
		self.file_out.set(self.file_in.get())
		self.save_path.set(os.path.split(filein)[0])
		Hstr='H' # for whatever reason, can't double nest quotations in an f-string :/
		self.brightness.set(f'{float(f"{np.mean(self.data[Hstr])+np.std(self.data[Hstr]):.3g}"):g}')
		self.threshold.set(f'{float(f"{np.mean(self.data[Hstr])+np.std(self.data[Hstr]):.3g}"):g}')
		self.comps_to_show_arr=list(range(len(self.data['H'])))
		self.init_plots()
		self.process_H_W()
	
	def load_config(self,filein=None):
		'''
		Loads .p file containing dict of parameters needed to create outputs. If display=True, sets GUI fields.
		'''
		if filein is None:
			filein=uiopen(title='Select pickle file for import',filetypes=[('pickle file','*.p'),('pickle file','*.pkl'),('pickle file','*.pickle')])
		if filein=='.':
			return
		with open(filein, "rb") as f:
			self.cfg=pickle.load(f)
			if self.display:
				for key,value in self.cfg.items():
					if self_fns[key] is 'entry':
						getattr(self,key).set(value)
					else:
						setattr(self,key,value)
				self.refresh_GUI()
			else:
				return self

	def refresh_GUI(self,event=None):
		#self.init_plots()
		if 'W_pp' in self.data:
			if self.frameslider.get() > len(self.data['H_pp'].T): # This (usually) occurs when the user crops the dataset
				self.frameslider.set(1)
			self.frameslider['to']=int(len(self.data['H_pp'].T)-1)
			self.imWH=self.Wax1.imshow((self.data['W_pp']@np.diag(self.data['H_pp'][:,self.frameslider.get()])@\
				self.cmap[:,:-1]*(255/self.cfg['brightness'])).reshape(self.data['W_shape'][0],self.data['W_shape'][1],3)\
				.clip(min=0,max=255).astype('uint8'))
			self.imW=self.Wax2.imshow((self.data['W_pp']@self.cmap[:,:-1]*255/np.max(self.data['W_pp'])).\
				reshape(self.data['W_shape'][0],self.data['W_shape'][1],3).clip(min=0,max=255).astype('uint8'))
			self.imW.axes.set_aspect('equal')
			self.imWH.axes.set_aspect('equal')
			self.canvas_W.draw()
			self.refresh_slider([])

		if 'H_pp' in self.data:
			Hstd=self.data['H_pp'].std()*3
			if self.offsetH.get():
				tmpH=self.data['H_pp'].T - repmat([w*Hstd for w in list(range(len(self.comps_to_show_arr)))],len(self.data['H_pp'].T),1)
			else:
				tmpH=self.data['H_pp'].T

			self.Hax1.cla()
			self.Hax1.plot(tmpH,linewidth=.5)
			for i,j in enumerate(self.Hax1.lines):
				j.set_color(self.cmap[i])
			if not self.offsetH.get():
				thresh_line=self.Hax1.plot(np.ones((len(self.data['H_pp'].T,)))*self.cfg['threshold'],linestyle='dashed',color='0',linewidth=1)
				zero_line=self.Hax1.plot(np.zeros((len(self.data['H_pp'].T,))),linestyle='dashed',color='.5',linewidth=1)
				self.legend=self.Hax1.legend((thresh_line[0],), ('Threshold',))
			self.Hax1.set_xlim(0, len(self.data['H_pp'].T))
			self.Hax1.set_ylim(np.min(tmpH), np.max(tmpH))
			if self.offsetH.get():
				self.Hax1.set(ylabel='Component #')
			else:
				self.Hax1.set(ylabel='Magnitude')

			if self.audio_analog.get():
				if len(self.data['H_pp'])>16:
					self.audio_analog.set(0)
					self.Hax2.imshow(self.data['H_fp'],interpolation='none',cmap=plt.get_cmap('gray'))
					self.message('Error: Analog audio is currently limited to 16 components.')
				else:
					# I do not understand why I have to do it this way
					tmp=self.Hax2.imshow(self.data['H_pp'],interpolation='none',cmap=plt.get_cmap('gray'))
					tmp.set_clim(0, np.max(self.data['H_pp']))
			else:
				self.Hax2.imshow(self.data['H_fp'],interpolation='none',cmap=plt.get_cmap('gray'))

			self.Hax2.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '{:.2g}'.format(x/self.cfg['fr'])))

			if len(self.comps_to_show_arr) > 12:
				yticks=np.arange(4,len(self.data['H_pp']),5)
				yticklabels=np.arange(4,len(self.data['H_pp']),5)
			else:
				yticks=np.arange(0,len(self.data['H_pp']),1)
				yticklabels=np.arange(0,len(self.data['H_pp']),1)

			if self.offsetH.get():
				self.Hax1.set(yticks=-yticks*Hstd,yticklabels=yticklabels)
			self.Hax2.set(yticks=yticks,yticklabels=yticklabels)
			self.Hax2.axes.set_aspect('auto')
			self.canvas_H.draw()

	def process_H_W(self):
		'''
		Core function of pyanthem. Applies all cfg settings to dataset, and 
		creates the note dict used for synthesis. Automatically calls 
		refresh_GUI() if display=True
		'''
		if self.display:
			self.self_to_cfg()
			self.message('Updating...')
			self.update()
		if self.cfg['comps_to_show']=='all':
			self.comps_to_show_arr=list(range(len(self.data['H'])))
		# regex expression which lazily checks for a bracketed expression containing numbers, colons and commas.
		elif re.match('^\[[0-9,: ]*\]$',self.cfg['comps_to_show']) is not None:
			# This is a magic function which transforms bracketed string arrays to actual numpy arrays.
			# Example: '[1,3,5:8]' --> array([1,3,5,6,7])
			self.comps_to_show_arr=eval('np.r_'+self.cfg['comps_to_show']) 
		else:
			self.message('For \'components to show\', please input indices with commas and colons enclosed by square brackets, or \'all\' for all components.')
			return
		
		self.data['H_pp']=self.data['H'][self.comps_to_show_arr,int(len(self.data['H'].T)*self.cfg['start_percent']/100):int(len(self.data['H'].T)*self.cfg['end_percent']/100)]
		self.data['H_pp']=self.data['H_pp']+self.cfg['baseline']
		if 'W' in self.data:
			self.data['W_pp']=self.data['W'][:,self.comps_to_show_arr]
		
		# make_keys()
		self.keys,i=[],0
		while len(self.keys) < len(self.data['H_pp']):
			self.keys.extend([k+i+key_opts[self.cfg['key']]+octave_add_opts[self.cfg['octave_add']] for k in scale_keys[self.cfg['scale_type']]])
			i+=12
		self.keys=self.keys[:len(self.data['H_pp'])]
		self.keys=[min(k,127) for k in self.keys] # Notes cannot be higher than 127

		# Making note dict
		true_fr=self.cfg['fr']*self.cfg['speed']/100
		upsample=1000
		ns=int(len(self.data['H_pp'].T)*upsample/true_fr)
		t1=np.linspace(0,len(self.data['H_pp'].T)/self.cfg['fr'],len(self.data['H_pp'].T))
		t2=np.linspace(0,len(self.data['H_pp'].T)/self.cfg['fr'],ns)
		nchan=len(self.data['H_pp'])
		Hmax=np.max(self.data['H_pp'])
		self.data['H_fp']=np.zeros(np.shape(self.data['H_pp']))
		self.data['H_pp'][self.data['H_pp'] < 0]=0
		self.data['H_pp'][:,-1]=0
		self.data['H_pp'][:,0]=0
		self.nd=[]
		for i in range(nchan):
			H_rs=interp1d(t1,self.data['H_pp'][i,:])(t2)
			H_b=H_rs.copy()
			H_b[H_b<self.cfg['threshold']]=0
			H_b[H_b>=self.cfg['threshold']]=1
			TC=np.diff(H_b)
			st=np.argwhere(TC==1)
			en=np.argwhere(TC==-1)
			st=np.ndarray.flatten(st).tolist()
			en=np.ndarray.flatten(en).tolist()
			for j in range(len(st)):
				mag=np.max(H_rs[st[j]:en[j]])
				self.data['H_fp'][i,int(st[j]*true_fr/upsample):int(en[j]*true_fr/upsample)]=mag
				# Type, time, note, velocity
				self.nd.append(['note_on',st[j],self.keys[i],int(mag * 127 / Hmax)])
				self.nd.append(['note_off',en[j],self.keys[i],int(mag * 127 / Hmax)])
			self.nd=sorted(self.nd, key=lambda x: x[1])
		# Colormap
		cmap=getattr(cmaps,self.cfg['cmapchoice'])
		self.cmap=cmap(np.linspace(0,1,len(self.data['H_pp'])))
		if self.display:
			self.refresh_GUI()

	def refresh_slider(self,event):
		'''
		
		'''
		self.imWH.set_data((self.data['W_pp']@np.diag(self.data['H_pp'][:,self.frameslider.get()])@self.cmap[:,:-1]*(255/self.cfg['brightness'])).reshape(self.data['W_shape'][0],self.data['W_shape'][1],3).clip(min=0,max=255).astype('uint8'))
		self.canvas_W.draw()

	def preview_notes(self):
		'''
		Previews the self.keys list audibly and visually simultaneously.
		'''
		self.process_H_W()
		self.message('Previewing notes...')
		fn_font=os.path.join(os.path.dirname(os.path.abspath(__file__)),'anthem_soundfonts','font.sf2')
		fn_midi=os.path.join(os.path.dirname(os.path.abspath(__file__)),'preview.mid')
		fn_wav=os.path.join(os.path.dirname(os.path.abspath(__file__)),'preview.wav')
		if get_init() is None: # Checks if pygame has initialized audio engine. Only needs to be run once per instance
			pre_init(fs, -16, 2, 1024)
			init()
			set_num_channels(128) # We will never need more than 128...
		mid=mido.MidiFile()
		track=mido.MidiTrack()
		mid.tracks.append(track)
		mid.ticks_per_beat=1000
		track.append(mido.MetaMessage('set_tempo', tempo=int(1e6)))
		track.append(mido.Message('program_change', program=sound_presets[self.cfg['sound_preset']], time=0))
		for i in range(len(self.keys)):
			track.append(mido.Message('note_on', note=self.keys[i], velocity=100, time=250))
			track.append(mido.Message('note_off', note=self.keys[i], time=250))
		track.append(mido.Message('note_off', note=self.keys[i], time=500))
		mid.save(fn_midi)
		cmd='fluidsynth -ni -F {} -r {} {} {} '.format(fn_wav,fs,fn_font,fn_midi)
		print(cmd)
		os.system(cmd)
		music.load(fn_wav)
		for i in range(len(self.keys)):
			t=time.time()
			self.imW.remove()
			Wtmp=self.data['W_pp'][:,i]
			cmaptmp=self.cmap[i,:-1]
			self.imW=self.Wax2.imshow((Wtmp[:,None]@cmaptmp[None,:]*255/np.max(self.data['W_pp'])).reshape(self.data['W_shape'][0],self.data['W_shape'][1],3).clip(min=0,max=255).astype('uint8'))
			self.canvas_W.draw()
			self.update()
			if i==0:
				music.play(0)
			time.sleep(.5-np.min(((time.time()-t),.5)))
		time.sleep(1)
		try:
			os.remove(fn_midi)
			os.remove(fn_wav)
		except:
			pass
		self.refresh_GUI()
		
	def write_audio(self):
		'''
		Writes audio either from .sf2 using fluidsynth or into raw file.
		'''
		if not self.check_data():
			return
		self.process_H_W()
		fn_midi=os.path.join(self.cfg['save_path'],self.cfg['file_out'])+'.mid'
		fn_wav=os.path.join(self.cfg['save_path'],self.cfg['file_out'])+'.wav'
		fn_font=os.path.join(os.path.dirname(os.path.abspath(__file__)),'anthem_soundfonts','font.sf2')
		mid=mido.MidiFile()
		track=mido.MidiTrack()
		mid.tracks.append(track)
		mid.ticks_per_beat=1000
		track.append(mido.MetaMessage('set_tempo', tempo=int(1e6)))
		if not self.cfg['audio_analog']:
			track.append(mido.Message('program_change', program=sound_presets[self.cfg['sound_preset']], time=0))
			current_time=0
			for i in range(len(self.nd)):
				track.append(mido.Message(self.nd[i][0], time=int(self.nd[i][1]-current_time), note=self.nd[i][2], velocity=self.nd[i][3]))
				current_time=self.nd[i][1]
			mid.save(fn_midi)
		else:
			true_fr=self.cfg['fr']*self.cfg['speed']/100
			upsample=100
			ns=int(len(self.data['H_pp'].T)*upsample/true_fr)
			t1=np.linspace(0,len(self.data['H_pp'].T)/self.cfg['fr'],len(self.data['H_pp'].T))
			t2=np.linspace(0,len(self.data['H_pp'].T)/self.cfg['fr'],ns)
			nchan=len(self.data['H_pp'])
			Hmax=np.max(self.data['H_pp'])
			H_rs=np.zeros((nchan,ns))
			ticks=int(mid.ticks_per_beat/upsample)
			for i in range(nchan):
				H_rs[i,:]=interp1d(t1,self.data['H_pp'][i,:])(t2)
			for chan in range(nchan):
				track.append(mido.Message('program_change', channel=chan, program=sound_presets[self.cfg['sound_preset']], time=0))
				track.append(mido.Message('note_on', channel=chan, note=self.keys[chan], time=0))
				track.append(mido.Message('control_change', channel=chan, control=7, value=0, time=0))
			for i in range(ns):
				for chan in range(nchan):
					if chan==0:
						realticks=ticks
					else:
						realticks=0
					track.append(mido.Message('control_change', channel=chan, control=7, value=int(H_rs[chan,i]*127/Hmax), time=realticks))
			for chan in range(nchan):
				track.append(mido.Message('note_off', channel=chan, note=self.keys[chan], time=0))
			mid.save(fn_midi)
		os.system('fluidsynth -ni -F {} -r {} {} {}'.format(fn_wav,fs,fn_font,fn_midi))
		self.message(f'Audio file written to {self.cfg["save_path"]}')

	def write_video(self):
		'''
		Writes video file using self.data['H_pp'] and self.data['W_pp'] using ffmpeg. 
		We avoid using opencv because it is very slow in a conda environment
		http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
		'''
		if not self.check_data():
			return

		self.process_H_W()
		fn_vid=os.path.join(self.cfg['save_path'],self.cfg['file_out'])+'.mp4'
		v_shape=self.data['W_shape'][::-1][1:] # Reverse because ffmpeg does hxw
		command=[ 'ffmpeg',
			'-loglevel', 'warning', # Prevents excessive messages
			'-hide_banner',
			'-y', # Auto overwrite
			'-f', 'image2pipe',
			'-vcodec','png',
			'-s', '{}x{}'.format(v_shape[0],v_shape[1]),
			'-r', str(self.cfg['fr']*self.cfg['speed']/100),
			'-i', '-', # The input comes from a pipe
			'-an', # Tells FFMPEG not to expect any audio
			'-q:v','2', # Quality
			'-vcodec', 'mpeg4',
			fn_vid]
		pipe=sp.Popen(command, stdin=sp.PIPE)
		nframes=len(self.data['H_pp'].T)
		for i in range(nframes):
			frame=(self.data['W_pp']@np.diag(self.data['H_pp'][:,i])@self.cmap[:,:-1]*(255/self.cfg['brightness'])).reshape(self.data['W_shape'][0],self.data['W_shape'][1],3).clip(min=0,max=255).astype('uint8')
			im=Image.fromarray(frame)
			im.save(pipe.stdin, 'PNG')
			if self.display and i%10==0:
				self.status['text']=f'Writing video file, {i} out of {nframes} frames written'
				self.update()
		pipe.stdin.close()
		pipe.wait()
		self.message(f'Video file written to {self.cfg["save_path"]}')
		return self
	
	def merge(self):
		'''
		Merges video and audio with ffmpeg
		'''
		fn=os.path.join(self.cfg['save_path'],self.cfg['file_out'])
		print('hey')
		cmd='ffmpeg -hide_banner -loglevel warning -y -i {} -i {} -c:a aac -map 0:v:0 -map 1:a:0 {}'.format(fn+'.mp4',fn+'.wav',fn+'_AV.mp4')
		os.system(cmd)
		self.message(f'A/V file written to {self.cfg["save_path"]}')
		return self
	
	def write_AV(self):
		'''
		Runs full write and merge
		'''
		self.write_video()
		self.write_audio()
		self.merge()
		if not self.display:
			return self

	def cleanup(self):
		'''
		Tries to remove any files that are video or audio only.	
		'''
		fn=os.path.join(self.cfg['save_path'],self.cfg['file_out'])
		try: 
			os.remove(fn+'.mp4')
		except OSError: 
			pass
		try: 
			os.remove(fn+'.wav')
		except OSError: 
			pass
		try: 
			os.remove(fn+'.mid')
		except OSError: 
			pass
		self.message(f'A/V only videos removed')
		return self

	def edit_save_path(self):
		self.save_path.set(fd.askdirectory(title='Select a directory to save output files',initialdir=self.cfg['save_path']))

	def initGUI(self):
		'''
		Initialize GUI fields, labels, dropdowns, etc.
		'''
		self.winfo_toplevel().title('pyanthem v{}'.format(pkg_resources.require("pyanthem")[0].version))
		self.protocol("WM_DELETE_WINDOW", self.quit)

		# StringVars
		self.file_in=init_entry(None)
		self.file_out=init_entry(None)
		self.save_path=init_entry(None)
		self.speed=init_entry(100)
		self.fr=init_entry(0)
		self.start_percent=init_entry(0)
		self.end_percent=init_entry(100)
		self.baseline=init_entry(0)
		self.brightness=init_entry(0)
		self.threshold=init_entry(0)
		self.octave_add=init_entry('2')
		self.scale_type=init_entry('Maj. 7 (4/oct)')
		self.key=init_entry('C')
		self.sound_preset=init_entry('Piano')
		self.comps_to_show=init_entry('all')
		self.cmapchoice=init_entry('jet')
		
		# Labels
		Label(text='',font='Helvetica 1 bold').grid(row=0,column=0) # Just to give a border around Seperators
		Label(text='File Parameters',font='Helvetica 14 bold').grid(row=1,column=1,columnspan=2,sticky='WE')
		Label(text='Movie Parameters',font='Helvetica 14 bold').grid(row=1,column=3,columnspan=2,sticky='WE')
		Label(text='Audio Parameters',font='Helvetica 14 bold').grid(row=1,column=5,columnspan=2,sticky='WE')
		Label(text='Input Filename').grid(row=2, column=1,columnspan=2,sticky='W')
		Label(text='Output Filename').grid(row=4, column=1,columnspan=2,sticky='W')
		Label(text='Save Path').grid(row=6, column=1,columnspan=1,sticky='W')
		Label(text='Speed (%)').grid(row=2, column=3, sticky='E')
		Label(text='Start (%)').grid(row=3, column=3, sticky='E')
		Label(text='End (%)').grid(row=4, column=3, sticky='E')
		Label(text='Baseline').grid(row=5, column=3, sticky='E')
		Label(text='Max brightness').grid(row=6, column=3, sticky='E')
		Label(text='Colormap').grid(row=7, column=3, sticky='E')
		Label(text='Threshold').grid(row=2, column=5, sticky='E')
		Label(text='Octave').grid(row=3, column=5, sticky='E')
		Label(text='Scale Type').grid(row=4, column=5, sticky='E')
		Label(text='Key').grid(row=5, column=5, sticky='E')
		Label(text='Audio format').grid(row=6, column=5, sticky='E')

		# Messages
		self.status=Message(text='Welcome to pyanthem v{}!'.\
			format(pkg_resources.require("pyanthem")[0].version),bg='white',fg='black',width=450)
		self.status.grid(row=9, column=2, columnspan=5, sticky='NESW')
		self.status['anchor']='nw'

		# Entries
		Entry(textvariable=self.file_in).grid(row=3, column=1,columnspan=2,sticky='W')
		Entry(textvariable=self.file_out).grid(row=5, column=1,columnspan=2,sticky='W')
		Entry(textvariable=self.save_path,width=17).grid(row=7, column=1,columnspan=2,sticky='EW')
		Entry(textvariable=self.speed,width=7).grid(row=2, column=4, sticky='W')
		Entry(textvariable=self.start_percent,width=7).grid(row=3, column=4, sticky='W')
		Entry(textvariable=self.end_percent,width=7).grid(row=4, column=4, sticky='W')
		Entry(textvariable=self.baseline,width=7).grid(row=5, column=4, sticky='W')
		Entry(textvariable=self.brightness,width=7).grid(row=6, column=4, sticky='W')
		self.threshold_entry=Entry(textvariable=self.threshold,width=7)
		self.threshold_entry.grid(row=2, column=6, sticky='W')

		# Buttons
		Button(text='Edit',command=self.edit_save_path,width=5).grid(row=6, column=2)
		Button(text='Preview',width=11,command=self.preview_notes).grid(row=7, column=5,columnspan=1)
		self.update_button=Button(text='Update',width=7,font='Helvetica 14 bold',command=self.process_H_W)
		self.update_button.grid(row=9, column=1,columnspan=1)
		
		# Option Menus
		self.octave_add_menu=OptionMenu(self,self.octave_add,*octave_add_opts.keys())
		self.octave_add_menu.config(width=7)
		self.octave_add_menu.grid(row=3, column=6, sticky='W')
		self.scale_type_menu=OptionMenu(self,self.scale_type,*scale_keys.keys())
		self.scale_type_menu.config(width=11,font=(self.default_font,(8)))
		self.scale_type_menu.grid(row=4, column=6, sticky='W')
		self.key_menu=OptionMenu(self,self.key,*key_opts.keys())
		self.key_menu.config(width=7)
		self.key_menu.grid(row=5, column=6, sticky='W')
		self.sound_preset_menu=OptionMenu(self,self.sound_preset,*sound_presets.keys())
		self.sound_preset_menu.config(width=7)
		self.sound_preset_menu.grid(row=6, column=6, sticky='W')
		
		# Combo box
		self.cmapchooser=Combobox(self,textvariable=self.cmapchoice,width=5)
		self.cmapchooser['values']=cmaps_opts
		self.cmapchooser['state']='readonly'
		self.cmapchooser.grid(row=7, column=4, sticky='W')
		self.cmapchooser.current()
		self.cmap=[]

		# Checkbox
		self.audio_analog=IntVar()
		self.audio_analog.set(0)
		Checkbutton(self, text="Analog",command=self.process_H_W,variable=self.audio_analog).grid(row=7, column=6,columnspan=1)

		# Menu bar
		menubar=Menu(self)
		filemenu=Menu(menubar, tearoff=0)
		filemenu.add_command(label="Load from .mat", command=self.load_GUI)
		filemenu.add_command(label="Load .cfg", command=self.load_config)
		filemenu.add_command(label="Quit",command=self.quit,accelerator="Ctrl+Q")

		savemenu=Menu(menubar, tearoff=0)
		savemenu.add_command(label="Audio", command=self.write_audio)
		savemenu.add_command(label="Video", command=self.write_video)
		savemenu.add_command(label="Merge A/V", command=self.merge)
		savemenu.add_command(label="Write A/V then merge", command=self.write_AV)
		savemenu.add_command(label="Cleanup", command=self.cleanup)

		cfgmenu=Menu(menubar, tearoff=0)
		cfgmenu.add_command(label="Save", command=self.dump_cfg)
		cfgmenu.add_command(label="View", command=self.view_cfg)

		debugmenu=Menu(menubar, tearoff=0)
		debugmenu.add_command(label="Query", command=self.query)

		menubar.add_cascade(label="File", menu=filemenu)
		menubar.add_cascade(label="Save", menu=savemenu)
		menubar.add_cascade(label="Config", menu=cfgmenu)
		menubar.add_cascade(label="Debug", menu=debugmenu)
		self.config(menu=menubar)

		# Seperators
		s_v=[[0,1,9],[2,1,8],[4,1,8],[6,1,9]]
		s_h=[[1,1,6],[1,2,6],[1,9,6],[1,10,6],[1,4,2],[1,6,2]]
		for sv in s_v:
			Separator(self, orient='vertical').grid(column=sv[0], row=sv[1], rowspan=sv[2], sticky='nse')
		for sh in s_h:
			Separator(self, orient='horizontal').grid(column=sh[0], row=sh[1], columnspan=sh[2], sticky='nwe')

		# Offset
		self.offsetH=IntVar()
		self.offsetH.set(1)

		# Bind shortcuts
		self.bind_all("<Control-q>", self.quit)
		self.bind_all("<Control-a>", lambda:[self.process_H_W(),self.refresh_GUI()])

	def init_plots(self):
		'''
		Initializes the plot areas. Is called every time update_GUI() is called.
		'''
		# H
		if 'H_pp' in self.data:
			self.figH=plt.Figure(figsize=(6,6), dpi=100, tight_layout=True)
			self.Hax1=self.figH.add_subplot(211)
			self.Hax2=self.figH.add_subplot(212)
			self.Hax1.set_title('Temporal Data (H)')
			self.Hax2.set_title('Audio Preview (H\')')
			self.canvas_H=FigureCanvasTkAgg(self.figH, master=self)
			self.canvas_H.get_tk_widget().grid(row=1,column=7,rowspan=30,columnspan=10)
			bg=self.status.winfo_rgb(self['bg'])
			self.figH.set_facecolor([(x>>8)/255 for x in bg])
			self.Hax1.spines['left'].set_visible(False)
			self.Hax1.spines['top'].set_visible(False)
			self.Hax1.spines['bottom'].set_visible(False)
			self.Hax1.spines['right'].set_visible(False)
			self.Hax1.yaxis.tick_right()
			self.Hax1.yaxis.set_label_position("right")
			self.Hax1.tick_params(axis='x',which='both',bottom=False, top=False, labelbottom=False, right=False)
			self.Hax2.set(xlabel='time (sec)',ylabel='Component #')
			self.Hax2.spines['left'].set_visible(False)
			self.Hax2.spines['top'].set_visible(False)
			self.Hax2.spines['bottom'].set_visible(False)
			self.Hax2.spines['right'].set_visible(False)
			self.Hax2.yaxis.tick_right()
			self.Hax2.yaxis.set_label_position("right")
			# Checkbox
			Checkbutton(self, text="Offset H",command=self.refresh_GUI,variable=self.offsetH).grid(row=1,rowspan=1,column=16)
		
		try: 
			# If user previously loaded dataset with W, clear those 
			# plots and remove associated GUI elements.
			self.figW.clear()
			self.canvas_W.draw()
			self.frameslider.grid_forget()
			self.comps_to_show_label.grid_forget()
			self.comps_to_show_entry.grid_forget()
		except:
			pass

		if 'W_pp' in self.data:
			self.figW=plt.Figure(figsize=(6,3), dpi=100, constrained_layout=True)
			self.Wax1=self.figW.add_subplot(121)
			self.Wax2=self.figW.add_subplot(122)
			self.Wax1.set_title('Video Preview')
			self.Wax2.set_title('Spatial Data (W)')
			self.Wax1.axis('off')
			self.Wax2.axis('off')
			self.canvas_W=FigureCanvasTkAgg(self.figW, master=self)
			self.canvas_W.get_tk_widget().grid(row=11,column=1,rowspan=19,columnspan=6)
			self.figW.set_facecolor([(x>>8)/255 for x in bg])
			
			# Frameslider
			# frameslider
			self.frameslider=Scale(self, from_=1, to=2, orient=HORIZONTAL)
			self.frameslider.set(1)
			self.frameslider['command']=self.refresh_slider
			self.frameslider.grid(row=30, column=1, columnspan=3,sticky='EW')
			
		# comps_to_show
		self.comps_to_show_label=Label(text='Components:')
		self.comps_to_show_label.grid(row=30, column=5, columnspan=1, sticky='E')
		self.comps_to_show_entry=Entry(textvariable=self.comps_to_show,width=15,justify='center')
		self.comps_to_show_entry.grid(row=30, column=6, columnspan=1,sticky='W')
		
	def process_raw(self,data=None,n_clusters=None,frame_rate=None,save=False,file_in=None):
		'''
		Decomposes raw dataset. Can be used in two ways: as a part of the 
		GUI class for immediate processing (e.g. process_raw().write_AV()),
		or as a method to save a new dataset. 
		'''
		if data is None:
			file_in=uiopen(title='Select .mat file for import',filetypes=[('.mat files','*.mat')])
			if filein=='.':
				return
			dh, var=loadmat(file_in),whosmat(file_in)
			data=dh[var[0][0]]
		sh=data.shape
		if len(sh) != 3:
			self.message('ERROR: input dataset is not 3D.')
			return
		data=data.reshape(sh[0]*sh[1],sh[2])
		# Ignore rows with any nans
		nanidx=np.any(np.isnan(data), axis=1)
		data_nn=data[~nanidx] # nn=non-nan
		# k-means
		print('Performing k-means...',end='')
		if n_clusters is None:
			n_clusters=int(len(data)**.25) # Default k is the 4th root of the number of samples per frame (for 256x256, this would be 16)
			print(f'No num_clusters given. Defaulting to {n_clusters}...',end='')
		idx_nn=KMeans(n_clusters=n_clusters, random_state=0).fit(data_nn).labels_
		idx=np.zeros((len(data),))
		idx[nanidx==False]=idx_nn
		# TCs
		H=np.zeros((n_clusters,len(data.T)))
		for i in range(n_clusters):
			H[i,:]=np.nanmean(data[idx==i,:],axis=0)
		print('done.')
		# NNLS
		nnidx=np.where(~nanidx)[0]
		W=np.zeros((len(data),n_clusters))
		print('Performing NNLS...',end='')
		for i in range(len(nnidx)):
			W[nnidx[i],:]=nnls(H.T,data_nn[i,:])[0]
		# Sort bottom to top
		xc,yc=[],[]
		(X,Y)=np.meshgrid(range(sh[1]),range(sh[0]))
		for i in range(len(W.T)):
			Wtmp=W[:,i].reshape(sh[0],sh[1])
			xc.append((X*Wtmp).sum() / Wtmp.sum().astype("float"))
			yc.append((Y*Wtmp).sum() / Wtmp.sum().astype("float"))
		I=np.argsort(yc[::-1]) # Reverse orders from bottom to top
		W,H=W[:,I],H[I,:]
		print('done.')
		
		# Assign variables and save
		self.data={}
		self.data['H']=H
		self.data['W']=W.reshape(sh[0],sh[1],n_clusters)
		#self.data['W_shape']=self.data['W'].shape.flatten()
		if frame_rate==[]:
			self.data['fr']=10
			print('No fr given. Defaulting to 10')
		else:
			self.data['fr']=frame_rate
		if save:
			if file_in is None:
				file_in='data.mat'
			fn=file_in.replace('.mat','_decomp.mat')
			savemat(fn,self.data)
			self.message(f'Decomposed data file saved to {fn}')
		
		# Reshape W here, since any use of self from here would require a flattened W
		self.data['W']=self.data['W'].reshape(self.data['W'].shape[0]*self.data['W'].shape[1],self.data['W'].shape[2])
		return self

	def query(self):
		'''
		Prints self attribute to status field
		'''
		field=simpledialog.askstring("Input", "Query a root property",parent=self)
		try:
			self.status['text']=str(getattr(self,field))
		except:
			self.status['text']='Bad query.'

	def view_cfg(self):
		'''
		Prints cfg info to command line
		'''
		try:
			for key in self.cfg:
				print(str(key)+': '+str(self.cfg[key]))
		except:
			pass
		
	def help(self):
		print('To load a dataset:\npyanthem.load_data()\n\nTo load a cfg file:\npyanthem.load_config()\n\nTo write video:\npyanthem.write_video()\n\nTo write audio:\npyanthem.write_audio()')

if __name__=="__main__":
	run()

# self\.([a-z_]{1,14})\.get\(\)
# self\.cfg\[$1\]