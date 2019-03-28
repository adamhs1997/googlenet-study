import numpy
import tensorflow
from lucid.misc.io import show, load
from skimage.io import imsave
from PIL import Image

# Load in our image
img = numpy.array(Image.open("saliency5b2.png").resize((224,224))) / 255		# Adjust input saliency here
##img = load("saliency.png")

# Eliminate the low values--leave only high
img[img<0.4] = 0								# Adjust the dropout here

# Save off modified saliency
imsave("testing.png", img, plugin='pil', format_str='png')

###

# Get our overlay image
img_pastable = Image.open("image2.png")						# Adjust input image here
img_pastable = img_pastable.resize((224, 224), Image.ANTIALIAS)
img_overlay = Image.open("testing.png")
img_overlay = img_overlay.resize((224, 224))
img_overlay = img_overlay.convert("RGBA")
datas = img_overlay.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((0, 0, 0, 0))
    else:
        newData.append(item)

img_overlay.putdata(newData)

img_pastable.paste(img_overlay, (0,0), img_overlay)
img_pastable.save('overlaid', 'png', quality=100)

###

# Get the overlaid region out of the original image
x, y, z = (img != 0).nonzero()
print(x)
print(y)
print(z)
print(len(x), len(y), len(z))

# Iterate through the photo, using only the things ID'd as not highlighted
photo_chunk = numpy.zeros((224,224,3))
photo = numpy.array(Image.open("image2.png").resize((224,224), Image.ANTIALIAS)) / 255	# Adjust input image here
imsave("testing_chunks.png", photo, plugin='pil', format_str='png')
print(photo.shape)
print(photo_chunk.shape)

for idx, subx in enumerate(x):
  for i in range(0, 3):
    photo_chunk[subx, y[idx], i] = photo[subx, y[idx], i]
imsave("testing_chunk.png", photo_chunk, plugin='pil', format_str='png')


exit()

# Example on how to slice an image
print(img [7][15][0])
print(len(img))
print(img[7])
print(img[8])
