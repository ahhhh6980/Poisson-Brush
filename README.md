# Poisson-Brush

This project is a bit messy right now

This acts as a sort of clone tool, using the gradient domain to seamlessly paste
I deconvolve the gradient in the frequency domain to get back the original image, or the closest it can to the original image

Middle click to set an anchor point

left image is the destination, what you'll draw on
right image is the source, it will clone from the anchor point you set with middle click

left click to draw, right click to erase

on initial load, the fft wont do a good job, so you simply have to load the image again and you'll be set :)
