# Poisson-Brush

*Install requirements:* `pip install -r requirements.txt`

## This project is a bit messy right now, and it's definitely far from polished

**This acts as a sort of clone tool, using the gradient domain to seamlessly paste**
- I deconvolve the gradient in the frequency domain to get back the original image, or the closest it can to the original image

1 Middle click to set an anchor point

- left image is the destination, what you'll draw on
right image is the source, it will clone from the anchor point you set with middle click

2 left click to draw, right click to erase

- on initial load, the fft wont do a good job, so you simply have to load the image again and you'll be set :)

# To-Do:
- [X] Implement FFTW to replace NumPy for a faster integration of the image
- [ ] Compute the derivate, paste, and integral only in the region where pixels are copied, not the whole image
- [ ] Find better examples of ways to use this
- [ ] Make the UI prettier

![](https://ninja.dog/mQorTS.png)
