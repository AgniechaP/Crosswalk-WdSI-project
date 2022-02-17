# Crosswalk WdSI project
 Projekt końcowy z przedmiotu Wprowadzenie do Sztucznej Inteligencji - laboratorium
 
 Agnieszka Piórkowska 144548

 Politechnika Poznańska, 2021/2022
 
 Cel projektu: 
 
 Celem projektu było zaimplementowanie funkcjonalności rozpoznawania znaków przejścia dla pieszych (crosswalks) na podstawie przykładowej bazy znaków drogowych. 
 
** Wykonanie projektu:**

1) Pierwszym krokiem było podzielenie przykładowej bazy danych na dwie sekcje: Train oraz Test. W folderze "Train" znajdują się zdjęcia należące do bazy treningowej, natomiast folder "Train_xml" zawiera odpowiadające zdjęciom pliki .xml opisujące daną fotografię. Analogicznie jest dla folderów "Test" oraz "Test_xml" odzwierciedlających bazę testową,
2) W zawartości projektu znajdują się dwa skrypty umożliwiające konwersję danych z rozszezreniem .xml na dwa pliki .csv (Train.csv oraz Test.csv). Zapisane są w nich następujące dane: Amount (liczba znaków w obrębie jednej fotografii), Width, Height, Roi.X1 (zmienna xmin z pliku .xml), Roi.Y1 (ymin), Roi.X2 (xmax), Roi.Y2 (ymax), ClassId (klasa, do której należy dany znak drogowy), Path (ścieżka z nazwą pliku .png),
3) ClassId oznacza klasę, do której przyporządkowano dany znak. Jeśli w obrębie fotografii znajduje się znak "crosswalk" przydzielony jest do klasy 1, znak "stop" do klasy 2, znak "speedlimit" do klasy 3, natomiast "trafficlight" do klasy 4,
4) Dzięki tak utworzonym plikom możliwe było przystąpienie do realizacji docelowego założenia projektowego. W pliku "main.py" występuje szereg funkcji umożliwiający osiągnięcie wymaganego wyjścia,
5) Funkcja "load_data" służy załadowaniu poszczególnych danych z plików .csv i umiejscowieniu ich w liście z danymi. W celu poprawienia działania programu skorzystano także z "croppingu zdjęć", ktory umożliwił wyodrębnienie poszczególnych znaków drogowych z jednego zdjęcia, tak, by uzyskać lepszą skuteczność klasyfikacji znaków drogowych,
6) Funkcja "display_dataset_stats" wyświetla statystykę dotyczącą tego, ile znaków znajduje się w bazie treningowej, a ile w testowej, 
7) "learn_bovw" tworzy słownik (voc.npy) klasyfikujący zdjęcia (wykrycie kluczowych cech). Ekstrakcja lokalnych cech następuje przy pomocy detektora SIFT,
8) "extract_features" oblicza deskryptory dla obrazów, 
9) Funckja "train" ma na celu wytrenowanie modelu. Wyjściem tej funkcji jest model,
10) "predict" przyjmuje model oraz dane, predykuje etykiety przyjmowanych elementów,
11) Dokonanie ewaluacji następuje w funkcji "evaluate". Zaimplementowana jest tu funkcjonalność porównywania predykowanych etykiet z prawdziwymi. Dzięki funkcji można określić w jakim stopniu skuteczna jest klasyfikacja programu,
12) "display_data" - ta funkcja umożliwia wyświetlenie wymaganych danych. Jeśli na przyjętym zdjęciu znajduje się znak przejścia dla pieszych (crosswalk), program wypisuje nazwę zdjęcia, na którym wspomniany znak się znajduje. Linijkę niżej wyświetlona jest liczba wskazująca na ilość znaków znajdujących się w obrębie fotografii ze znakiem przejścia dla pieszych. Przykładowo, na zdjęciu "road309.png" znajdują się dwa znaki (konkretnie: speedlimit oraz crosswalk) i dlatego wyświetlana liczba wynosi 2. Pod spodem jest informacja o wymiarach obszaru, na którym znajduje się wykryty znak. Informacja zaczerpnięta jest ze wspomnianych wcześniej plików.
 
 
 
 **Bibliografia:**
 
 https://www.geeksforgeeks.org/convert-xml-to-csv-in-python/
 https://www.kaggle.com/andrewmvd/road-sign-detection
 https://pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
 
 Materiały z zajęć
