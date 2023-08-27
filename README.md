# Matching Goods
 The app created for matching goods purposes.
 There are big dataset with embedded features that represents goods from e-shop.
 Searching realized with faiss searching nearest neighbors and Catboost Classifier
 using as ranker that chooses 5 best candidates from faiss.
 The key metric is Accuracy@5 
 
 Before using the app you must load the data from:
 https://disk.yandex.ru/d/BBEphK0EHSJ5Jw
 After that you must put all the files into /data dir and run main.ipynb.
 This will create required files for app is working.

 There are dockerfile and docker-compose, so you could run it on your machine easily. 