nnotes = 30#len(data['H_pp'])
keys = []
i=0
key=0
oct_add=0
scaledata=[0,1,2,3,4,5,6,7,8,9,10,11]
while len(keys) < nnotes:
	keys.extend([k+i+key+oct_add*12 for k in scaledata])
	i+=12
keys = keys[:nnotes]
print(keys)