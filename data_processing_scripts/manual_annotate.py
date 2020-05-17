import kivy
import os
import PIL
from PIL import ImageDraw
import numpy as np
import json

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.config import Config

clip = '00000000'
ims = os.listdir(clip+'/align')
ims.sort()
imw = None
c = 1

m_pos = (0,0)

Window.size = (1800/2,1600/2)

def key_down(self,keyboard, keycode, text, modifiers):
	global m_pos
	global ims
	global c
	if keycode == 25:
		print(m_pos)
		x = m_pos[0]
		y = 1600 - m_pos[1]
		fpath = clip+'/robot_left/'+ims[c-1].replace('.png','.json')
		print('{}/{}'.format(c,len(ims)))
		with open(fpath,'w') as f:
			json.dump([x,y],f)
		'''
		im = PIL.Image.open('00000000/align/'+ims[c-1])
		draw = ImageDraw.Draw(im)
		r = 10
		draw.ellipse([x-r,y-r,x+r,y+r],outline='magenta')
		im.show()
		'''
		imw.source = clip+'/align/'+ims[c]
		c+=1

def on_motion(self, pos):
	global m_pos
	m_pos = pos


Window.bind(on_key_down=key_down)
Window.bind(mouse_pos=on_motion)

class MyApp(App):
	def build(self):
		global imw
		wimg = Image(source=clip+'/align/'+ims[0],size_hint=(None,None),pos=(0,0),width=1800,height=1600)
		imw = wimg
		return wimg

app = MyApp()
app.run()
