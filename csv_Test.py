# Importowanie potrzebnych bibliotek:
import xml.etree.ElementTree as Xet
import pandas as pd
import os
import glob2

cols = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
rows = []
Class_from_xml = 0
how_many_1 = 0
how_many_0 = 0
how_many_1_Test = 0
how_many_0_Test = 0
# Zliczanie liczby plikow w folderze Test_xml:
path, dirs, files = next(os.walk("./Test_xml"))
file_count = len(files)

#for j in range(0, file_count):
for filename in glob2.glob(os.path.join(path, '*.xml')):
   with open(os.path.join(os.getcwd(), filename), 'r') as f:
    xmlparse = Xet.parse(f)
    root = xmlparse.getroot()
    root2 = xmlparse.findall("object")
    Path = xmlparse.find("filename").text
    Width = xmlparse.find("size/width").text
    Height = xmlparse.find("size/height").text
    for i in root2:
        if i:
            RoiX1 = i.find("bndbox/xmin").text
            RoiY1 = i.find("bndbox/ymin").text
            RoiX2 = i.find("bndbox/xmax").text
            RoiY2 = i.find("bndbox/ymax").text
            Class_from_xml = i.find("name").text
            if Class_from_xml == 'crosswalk':
                ClassId = 1
                how_many_1 = how_many_1+1
            if Class_from_xml == 'stop':
                ClassId = 2
            if Class_from_xml == 'speedlimit':
                ClassId = 3
            if Class_from_xml == 'trafficlight':
                ClassId = 4
            #else:
            #    ClassId = 0
            #    how_many_0 = how_many_0+1

        rows.append([Width, Height, RoiX1, RoiY1, RoiX2, RoiY2, ClassId, 'Test/' + Path])
    df = pd.DataFrame(rows, columns=cols)

    # Tworzenie pliku .csv i zapis do niego:
    df.to_csv('Test.csv')
print("Liczba plikow w Test_xml:", file_count)
print("Liczba znakow crosswalks Test:", how_many_1)
print("Liczba znakow innych niz crosswalks Test:", how_many_0)