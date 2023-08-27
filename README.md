# Matching Goods
 The app created for matching goods purposes.
 There are big dataset with embedded features that represents goods from e-shop.
 Searching realized with faiss searching nearest neighbors and Catboost Classifier
 using as ranker that chooses 5 best candidates from faiss.
 The key metric is Accuracy@5 

 The best result is Accuracy@5 = 77.683 on validation set.
 
 Before using the app you must load the data from:
 https://disk.yandex.ru/d/BBEphK0EHSJ5Jw
 After that you must put all the files into /data dir and run main.ipynb.
 This will create required files for app is working.

 App can take two type of query:
 1. /knn - use this for searching 5 nearest goods for input vector of features.
 2. /add - use this to add new goods into DB.
 Parametres for query:
 params={"item": vec, "vec_name": vec_name})
 vec - string with 72 float features, delimiter is ","
 vec_name - string with the name of new goods

 There are dockerfile and docker-compose, so you could run it on your machine easily. 