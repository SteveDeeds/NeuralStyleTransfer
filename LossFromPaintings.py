import glob
import os
import random
import string

from PIL import Image

import NeuralStyleTransfer


def main():
    fileNames = glob.glob(os.path.join("paintings", "**", "*.jpg"))

    f = open("bestPainting.txt", "a")
    random.shuffle(fileNames)

    for fname in fileNames:
        img = Image.open(fname)
        if img.mode == 'RGB':
            _, loss = NeuralStyleTransfer.run_style_transfer("13921 Holyoke.jpg", fname, num_iterations=1)
            line = "{:.4e},".format(loss) + fname
            filtered_line = filter(lambda x: x in string.printable, line)
            f.writelines(filtered_line)
            f.writelines("\n")
            f.flush()
        else:
            print(fname + " apears to be a gray scale image")
    f.close()


if __name__ == "__main__":
    main()
