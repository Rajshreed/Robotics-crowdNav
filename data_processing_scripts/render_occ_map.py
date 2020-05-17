import json
import math
import os
from PIL import Image, ImageDraw

clip = '00000000'
n = 64

ims = os.listdir(clip+'/align')
ims.sort()

# MUST NORMALIZE ANGLE RESPECT TO TARGET


target = [1500,900]
scale = 0.1

valid_corners = [
	(450,400), # tl
	(1500,400), # tr
	(1500,1400), # br
	(450,1400)  # bl
]

def tform(point,center,angle,scale):
	out = [point[0]-center[0],point[1]-center[1]]

	rad = angle*math.pi/180
	s = math.sin(rad)
	c = math.cos(rad)
	out = [
		c*out[0] - s*out[1],
		s*out[0] + c*out[1]
	]

	out = [out[0]*scale+32,out[1]*scale+32]
	return out

def draw_o(data,out):
	draw = ImageDraw.Draw(out)
	pos = data['robot_pos']

	# get angle
	diff = [target[0]-pos[0],target[1]-pos[1]]
	ang = math.atan2(diff[1],diff[0])
	ang = -ang*180/math.pi
	print(ang)

	# draw default occupancy
	draw.rectangle([0,0,10000,10000],fill='white')

	# draw floor
	floor = []
	for x in valid_corners:
		floor.append(tuple(tform(x,pos,ang,scale)))
	draw.polygon(floor,fill='black')

	# draw humans
	for human in data['humans']:
		h2 = tform(human,pos,ang,scale)
		x = h2[0]
		y = h2[1]
		r = 4
		draw.ellipse([x-r,y-r,x+r,y+r],fill='white')

last_pos = None
for i,imname in enumerate(ims):
	print(imname)
	with open(clip+'/detections/'+imname.replace('.png','.json')) as f:
                data = json.load(f)

	with open(clip+'/detections/'+ims[i+1].replace('.png','.json')) as f:
                data2 = json.load(f)

	# render
	out_im = Image.new('RGB',(n,n))

	draw_o(data,out_im)

	out_im.save(clip+'/final_data_im/'+imname)

	# velocity/angular
	pos1 = data['robot_pos']
	pos2 = data2['robot_pos']

	s_x = (pos2[0] - pos1[0])**2
	s_y = (pos2[1] - pos1[1])**2
	s = math.sqrt(s_x + s_y)

	ang_v = data2['angle'] - data['angle']

	# angle correction, relative to goal
	diff = [target[0]-pos1[0],target[1]-pos1[1]]
	tang = math.atan2(diff[1],diff[0])
	tang = tang*180/math.pi
	tang += 90

	angle = tang - data['angle']

	data.update({
		'angle':angle,
		'speed': s,
		'rotation': ang_v
	})

	with open(clip+'/final_data_raw/'+imname.replace('.png','.json'),'w') as f:
		json.dump(data,f)

