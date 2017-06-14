import shutil

# 读取文件中图片信息根据性别分类图片到对应目录中
dirroot = "D:\\D\\本科毕设资料\\adience"
f = open(dirroot+"\\fold_frontal_4_data.txt","r")
i = 0

for line in f.readlines():
    line = line.split()
    dir = line[0]
    imgName = "coarse_tilt_aligned_face."+ line[2] +'.'+ line[1]
    if i > 0:
        if line[5]== "f":
            print("female")
            shutil.move(dirroot+'\\faces\\'+dir+'\\'+imgName, "data\\train\\female\\"+imgName)
        #     移动图片到female目录
        elif line[5]=="m":
            print("male")
            shutil.move(dirroot+'\\faces\\'+dir+'\\'+imgName, "data\\train\\male\\"+imgName)
        #     移动图片到male目录
        else:
            print("N")
            # 未识别男女
    i += 1
f.close()