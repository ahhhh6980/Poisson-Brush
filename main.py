# PoissonBrush
# Main File
# (C) 2022 by Jacob (ahhhh6980@gmail.com)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys, os, logging, datetime, pyfftw, psutil
import numpy as np
import cv2 as cv

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import tkinter.font as font
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps

script_dir = os.getcwd()

def gradient(img):
    # Pad with 0's to preserve basis
    img = np.pad(img, 3, mode='constant')
    # Apply blur for LoG
    #blur = cv.GaussianBlur(img,(3,3),0)
    #laplacian = cv.Laplacian(img,cv.CV_64F)
    h,w = img.shape
    kernel = np.pad(np.array(
        [[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]]
        ), [(0,h-3), (0,w-3)])
    laplacian = apply_kernel(img, kernel)
    # Return rounded int
    return np.rint(laplacian)

pyfftw.interfaces.cache.enable()
simd = pyfftw.simd_alignment
threadCount = psutil.cpu_count()
if threadCount < 6:
    threadCount = 2
else: threadCount = 6
print("SIMD Alignment =", simd)
print("Threads In Use =", threadCount)
global written_wisdom
written_wisdom = False
def apply_kernel(img, kernel):
    global written_wisdom
    
    # These are intialized to be empty, and aligned for SIMD utilization
    kernelF = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    imgF = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    newIMG = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    kernel = pyfftw.byte_align(kernel.astype(np.complex64), n=simd)
    img = pyfftw.byte_align(img.astype(np.complex64), n=simd)

    if not written_wisdom:
        written_wisdom = True
        print("This time will be slower, computing best method of applying the fft...", end="")

    # Create Fourier Transform object, and then execute it!
    a = pyfftw.FFTW(kernel, kernelF, direction='FFTW_FORWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    a.execute()

    # Create Fourier Transform object, and then execute it!
    b = pyfftw.FFTW(img, imgF, direction='FFTW_FORWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    b.execute()

    # Create Fourier Transform object, and then execute it!
    temp = pyfftw.byte_align((kernelF * imgF).astype(np.complex64), n=simd)
    c = pyfftw.FFTW(temp, newIMG, direction='FFTW_BACKWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    c.execute()

    # The real component is the part we want
    return newIMG.real 

def compute_greenF(h,w):
    diracF = pyfftw.empty_aligned((h,w), dtype='complex64', n=simd)
    laplaceF = pyfftw.empty_aligned((h,w), dtype='complex64', n=simd)

    # Padded Dirac kernel in Frequency domain
    dirac = np.pad(np.array([[0,0,0],[0,1,0],[0,0,0]]), [(3,w-6),(3,h-6)], mode='constant').astype(np.complex64)
    diracF_Object = pyfftw.FFTW(dirac, diracF, direction='FFTW_FORWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    diracF_Object.execute()

    # Padded Laplacian kernel in Frequency domain
    laplace = np.pad(np.array([[0,1,0],[1,-4,1],[0,1,0]]), [(3,w-6),(3,h-6)], mode='constant').astype(np.complex64)
    laplaceF_Object = pyfftw.FFTW(laplace, laplaceF, direction='FFTW_FORWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    laplaceF_Object.execute()

    # Greens Function in Frequency domain
    return (diracF / (laplaceF + (1 / (w * h))))

global greenF_computed
greenF_computed = False
greenF = []
def reconstruct(img, oimg):
    global greenF_computed, greenF
    w,h = img.shape
    if not greenF_computed:
        greenF_computed = True
        greenF = compute_greenF(h,w)
    
    imgF = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    x = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)

    imgTemp = img.copy().astype(np.complex64)

    # Image in Frequency Domain
    imgF_Object = pyfftw.FFTW(imgTemp, imgF, direction='FFTW_FORWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    imgF_Object.execute()

    # Deconvolve img using greens function
    resultF = imgF * greenF

    # Inverse transform
    resultObject = pyfftw.FFTW(resultF, x, direction='FFTW_BACKWARD', 
        axes=(0,1),threads=threadCount, flags=('FFTW_MEASURE',))
    resultObject.execute()

    x = x.real
    x = x - x.min()
    
    # Color/contrast correction
    nx = ((x - np.average(x)) * (np.std(oimg) / np.std(x))) + np.average(oimg)
    nx = nx - nx.min()
    nx = nx / nx.max() * oimg.max()

    print("A",nx.min(), nx.max())
    nx = np.rint(nx)
    # Return rounded int
    return nx

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        img_path = script_dir+'/lenna.png'
        greenF = ""
        greenFRun = False
        self.iterat = 0
        self.circle = 0
        self.circle2 = 0
        self.v = 3
        self.anchorX = 10
        self.anchorY = 10
        self.drawX = 0
        self.drawY = 0
        self.srcW = 0
        self.srcH = 0
        self.lastDistance = [-1, -1]
        self.clickStartX = 0
        self.clickStartY = 0
        self.attributes('-alpha', 0.0) 

        self.title("Poisson Brush")
        self.geometry("1024x812")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.title("Poisson Brush")
        self.frame = tk.Frame(self)
        self.frame.grid(row=0, column=0, sticky="NEWS")
        self.frame.rowconfigure(0, weight=2)
        for i in range(4):
            self.frame.columnconfigure(i, weight=1)
        self.frame.rowconfigure(1, weight=0)
        self.frame.rowconfigure(2, weight=1)
        self.width = 1024
        self.height = 812

        self.update()

        self.canvas = tk.Canvas(self.frame, borderwidth=0, highlightthickness=0, height=512, relief=tk.FLAT, bd=-2)
        self.canvas.grid(row = 0, columnspan=2, pady = 0, padx = 0, sticky="NESW")
        self.canvas.bind("<Key>", self.key)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-2>", self.anchor)
        self.canvas.bind("<B1-Motion>", self.leftDown)
        self.canvas.bind("<B3-Motion>", self.rightDown)
        self.canvas.bind("<Motion>", self.motion)
        self.bind("<Delete>", self.resetImage)
        self.bind("<Lock-KeyPress>", self.toggleHold)
        self.bind("<Return>", self.updateImage)
        self.bind("<space>", self.resizeCanvas)
        self.bind("<Button-4>", self.vUp)
        self.bind("<Button-5>", self.vDown)
        self.hold = True
        self.canvas2 = tk.Canvas(self.frame, borderwidth=0, highlightthickness=0, height=512, relief=tk.FLAT, bd=-2)
        self.canvas2.grid(row = 0, column=2, columnspan=2, pady = 0, padx = 0, sticky="NESW")

        openImgBtn = tk.Button(self.frame, highlightthickness=0, relief=tk.FLAT, height=2, bg="light blue", text="Open Image", command=self.openImageCanvas)  
        openImgBtn.grid(row = 1, column=0, pady = 0, padx = 0, sticky="NESW")

        saveImgBtn = tk.Button(self.frame, highlightthickness=0, relief=tk.FLAT, height=2, bg="light green", text="Save Image", command=self.saveImage)  
        saveImgBtn.grid(row = 1, column=1, pady = 0, padx = 0, sticky="NESW")

        updateImgBtn = tk.Button(self.frame, highlightthickness=0, relief=tk.FLAT, height=2, bg="pink", text="Update Image", command=self.updateImage)  
        updateImgBtn.grid(row = 1, column=2, pady = 0, padx = 0, sticky="NESW")

        openSrcBtn = tk.Button(self.frame, highlightthickness=0, relief=tk.FLAT, height=2, bg="light blue", text="Open Source", command=self.openImageCanvas2)  
        openSrcBtn.grid(row = 1, column=3, pady = 0, padx = 0, sticky="NESW")

        self.logs = scrolledtext.ScrolledText(self.frame, highlightthickness=0, pady = 0, padx = 0, relief=tk.FLAT, height=12, wrap = "word")
        self.logs.grid(row=2, columnspan=4,rowspan=2, sticky="NESW")
        self.logs.configure(state="disabled")
        self.frame.grid_rowconfigure(2, minsize=10, weight=1)
        #self.logs.maxsize(1000,8)
        #sys.stdout = addwritemethod(self.logs)

    def openImage(self, widget, dest):
        w,h = widget.winfo_width(), widget.winfo_height()

        
        if dest:
            # Get image path
            self.img_path = filedialog.askopenfilename(initialdir=os.getcwd())

            # Save Image and original Image
            self.canvasImgs = [Image.open(self.img_path), Image.open(self.img_path)]
            self.canvasPixels = [self.canvasImgs[0].load(), self.canvasImgs[1].load()]

            # Compute gradients of image
            self.imgGradDest = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path))]]
            self.oimgGradDest = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path))]]

            # Display image towidget
            newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradDest])
            newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
            self.setTheImage(Image.fromarray(newChannels), self.canvas)
        else:
            # Get image path
            self.img_path2 = filedialog.askopenfilename(initialdir=os.getcwd())
            # Save Image and original Image
            self.canvas2Imgs = [Image.open(self.img_path2), Image.open(self.img_path2)]
            self.canvas2Pixels = [self.canvas2Imgs[0].load(), self.canvas2Imgs[1].load()]
            
            # Compute gradients of image
            self.imgGradSrc = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path2))]]
            self.oimgGradSrc = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path2))]]

            self.srcH, self.srcW = self.imgGradSrc[0][0].shape

            # Display image towidget
            newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradSrc])
            newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
            self.setTheImage(self.canvas2Imgs[0], self.canvas2)


        module_logger.info(str(datetime.datetime.now())+":\n\tGradients Computed")

    def openImageCanvas(self):
        self.openImage(self.canvas, 1)

    def openImageCanvas2(self):
        self.openImage(self.canvas2, 0)

    def resizeCanvas(self, event):
        newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradDest])
        newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
        self.setTheImage(Image.fromarray(newChannels), self.canvas)
        self.setTheImage(self.canvas2Imgs[0], self.canvas2)

    def resetImage(self, event):
        print(event)
        # Save Image and original Image
        self.canvasImgs = [Image.open(self.img_path), Image.open(self.img_path)]
        self.canvasPixels = [self.canvasImgs[0].load(), self.canvasImgs[1].load()]

        # Compute gradients of image
        self.imgGradDest = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path))]]
        self.oimgGradDest = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path))]]

        # Display image towidget
        newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradDest])
        newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
        self.setTheImage(Image.fromarray(newChannels), self.canvas)

        # Save Image and original Image
        self.canvas2Imgs = [Image.open(self.img_path2), Image.open(self.img_path2)]
        self.canvas2Pixels = [self.canvas2Imgs[0].load(), self.canvas2Imgs[1].load()]
        
        # Compute gradients of image
        self.imgGradSrc = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path2))]]
        self.oimgGradSrc = [[gradient(e), e] for e in [*cv.split(cv.imread(self.img_path2))]]

        self.srcH, self.srcW = self.imgGradSrc[0][0].shape

        # Display image towidget
        newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradSrc])
        newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
        self.setTheImage(self.canvas2Imgs[0], self.canvas2)

        module_logger.info(str(datetime.datetime.now())+":\n\tReset Image")
        
    def setTheImage(self, img, widget):
        w,h = widget.winfo_width(), widget.winfo_height()
        img2 = (img).resize((w,h))
        img1 = ImageTk.PhotoImage(img2)
        widget.create_image(w/2,h/2, image=img1)
        widget.image=img1
        

    def saveImage(self):
        img_path = filedialog.asksaveasfilename(initialdir=os.getcwd())
        newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradDest])
        newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
        cv.imwrite(img_path, newChannels)
        module_logger.info(str(datetime.datetime.now())+":\n\tSaved Image To "+img_path)
    
    def key(self, event):
        module_logger.info(event.char)

    def leftDown(self, event):
        self.drawOnCanvas(event, 0)
    def rightDown(self, event):
        self.drawOnCanvas(event, 1)

    def click(self, event):
        if self.hold == True:
            self.clickStartX, self.clickStartY = event.x, event.y
        module_logger.info(str(datetime.datetime.now())+":\n\tClick + Drag started from "+str(self.clickStartX)+","+str(self.clickStartY))

    def toggleHold(self, event):
        self.hold = not self.hold
        module_logger.info(str(datetime.datetime.now())+":\n\tToggled Hold to " + str(self.hold))

    def anchor(self, event):
        self.anchorX, self.anchorY = event.x, event.y
        module_logger.info(str(datetime.datetime.now())+":\n\tAnchor Set At "+str(self.anchorX)+","+str(self.anchorY))

    def updateImage(self, event=None):
        newChannels = cv.merge([reconstruct(e[0], e[1]).astype(np.float32) for e in self.imgGradDest])
        newChannels = cv.cvtColor(newChannels, cv.COLOR_BGR2RGB).astype(np.uint8)
        self.setTheImage(Image.fromarray(newChannels), self.canvas)

    def drawOnCanvas(self, event, t):
        self.iterat += 1
        self.motion(event)
        self.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        print(cw,ch)
        w, h = self.canvasImgs[0].size
        #x, y = event.x-50 + (w / 4),event.y-10 + (h / 4)

        x =  int(np.floor((event.x / cw) * w))
        y =  int(np.floor((event.y / ch) * h))

        mx = int(np.floor((self.clickStartX / cw) * w  ) - x)
        my = int(np.floor((self.clickStartY / ch) * h  ) - y)

        ax = int(np.floor((self.anchorX / cw) * w) )
        ay = int(np.floor((self.anchorY / ch) * h) )

        v = self.v
        # for i in range(v+1):
        #     for j in range(v+1):
        #         ix =np.floor(i-(v//2) + x)
        #         iy =np.floor(j-(v//2) + y)
        #         if (ix >= 0 and ix < w and
        #             iy >= 0 and iy < h):
        
        a,b = ax - mx - ((self.v // 2) + 1), ay - my - 3
        a2 = int(np.floor((a / w) * self.srcW))
        b2 = int(np.floor((b / w) * self.srcH))
        m = circMask((2 * v), (2 * v), None, v)
        print(True in m)
        if t:
            for i in range(3):
                p1 = np.copy(self.imgGradDest[i][0][y-v:y+v, x-v:x+v])
                p2 = np.copy(self.oimgGradDest[i][0][y-v:y+v, x-v:x+v])
                p2[~m] = p1[~m]
                self.imgGradDest[i][0][y-v:y+v, x-v:x+v] = p2 
            self.canvas2Pixels[0][a,b] = self.canvasPixels[1][a,b]
        else:
            for i in range(3):
                p1 = np.copy(self.imgGradDest[i][0][y-v:y+v, x-v:x+v])
                p2 = np.copy(self.oimgGradSrc[i][0][b2-v:b2+v,a2-v:a2+v])
                p2[~m] = p1[~m]
                self.imgGradDest[i][0][y-v:y+v, x-v:x+v] = 0.5 * np.add(p1, p2)
            # self.canvas2Pixels[0][a,b] = (255,255,255)
   
        self.updateImage()
        #self.setTheImage(self.canvas2Imgs[0], self.canvas2)
        #module_logger.info(str(event.char) + str(mx) + "," + str(my))

    def vDown(self, event): 
        if self.v > 1:
            self.v -= 1
            self.motion(event)

    def vUp(self, event):
        self.v += 1
        self.motion(event)

    def motion(self, event):
        x, y = event.x, event.y
        self.canvas.delete(self.circle)
        radius = int(self.v * 1.5)
        self.circle = self.canvas.create_oval(x + radius, y + radius, 
                                              x - radius, y - radius, outline="black")

        self.canvas2.delete(self.circle2)
        radius = int(self.v * 1.5)
        self.circle2 = self.canvas2.create_oval(x + radius, y + radius, 
                                              x - radius, y - radius, outline="black")

def circMask(h, w, center=None, radius=None):
    Y, X = np.ogrid[:h, :w]
    dist = ((X - int(w/2))*(X - int(w/2))) + ((Y-int(h/2))*(Y-int(h/2)))
    mask = dist <= radius * radius
    return mask

class MyHandlerText(logging.StreamHandler):
    def __init__(self, textctrl):
        logging.StreamHandler.__init__(self) # initialize parent
        self.textctrl = textctrl

    def emit(self, record):
        msg = self.format(record)
        self.textctrl.config(state="normal")
        self.textctrl.insert("end", msg + "\n")
        self.flush()
        self.textctrl.config(state="disabled")
        self.textctrl.see("end")

module_logger = logging.getLogger(__name__)

def main():
    app = App()
    stderrHandler = logging.StreamHandler()  # no arguments => stderr
    module_logger.addHandler(stderrHandler)
    guiHandler = MyHandlerText(app.logs)
    module_logger.addHandler(guiHandler)
    module_logger.setLevel(logging.INFO)
    module_logger.info("Initialized: " + str(datetime.datetime.now()))
    app.mainloop()

if __name__ == '__main__':
    sys.exit(main())
