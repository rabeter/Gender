from PIL import Image
import glob, os

# 格式化图片像素224*224


def thumbnail(flie):
    i = 1
    for files in glob.glob(flie+'*.jpg'):
        filepath, filename = os.path.split(files)
        filterame, exts = os.path.splitext(filename)
        opfile = flie	# output path
        if (os.path.isdir(opfile) == False):
            os.mkdir(opfile)
        im = Image.open(files)
        im_ss = im.resize((224,224))
        print("Yes")
        im_ss.save(filename)
        i = i+1
    print(i,"张图片格式化完成")


if __name__ == '__main__':
    thumbnail('C:\\Users\\panni\\Desktop\\female\\')
