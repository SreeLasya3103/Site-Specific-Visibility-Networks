import tkinter as tk
from PIL import ImageTk, Image
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from dsets.Webcams import Webcams_cls
import os

DSET_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams'
LABELS_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/WebcamsGoodBadLists'

def writeLabel(file, label):
    file_list = os.path.join(LABELS_PATH, label+'.csv')
    file_list = open(file_list, "a")
    file_list.write(file+'\n')
    file_list.close()

def labelImage(imgIterator, gB, uB, bB, panel):
    try:
        rel_path = next(imgIterator)
        imgPath = os.path.join(DSET_PATH, rel_path)
    except:
        print("No more images!")
    
    print(imgPath)
    img = ImageTk.PhotoImage(Image.open(imgPath).resize((952,718)))
    panel.configure(image=img)
    panel.image = img
    
    def func(label):
        writeLabel(rel_path, label)
        labelImage(imgIterator, gB, uB, bB, panel)

    gB.configure(command=lambda:func('good'))
    uB.configure(command=lambda:func('unsure'))
    bB.configure(command=lambda:func('bad'))
    

with open(os.path.join(LABELS_PATH, 'files.csv')) as f:
    files = f.read().splitlines()
imgIter = iter(files)

m = tk.Tk()

imgPanel = tk.Label(m, image=None)

buttonsFrame = tk.Frame(m)
buttonsFrame.pack()
goodButton = tk.Button(buttonsFrame, text='Good')
goodButton.grid(row=0, column=0)
unsureButton = tk.Button(buttonsFrame, text='Unsure')
unsureButton.grid(row=0, column=1)
badButton = tk.Button(buttonsFrame, text='Bad')
badButton.grid(row=0, column=2)

imgPanel.pack()

def on_closing():
    with open(os.path.join(LABELS_PATH, 'files.csv'), 'w') as f:
        for file in imgIter:
            f.write(file+'\n')

    m.destroy()

m.protocol("WM_DELETE_WINDOW", on_closing)

labelImage(imgIter, goodButton, unsureButton, badButton, imgPanel)
m.mainloop()
