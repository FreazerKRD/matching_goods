import requests
import numpy as np

def main():
    vec = np.random.random(72)
    s = ','.join([str(el) for el in vec])
    
    r = requests.get("http://localhost:8031/knn", params={"item": s})

    if r.status_code == 200:
        print(r.json())
    else:
        print(r.status_code)

if __name__ == "__main__":
    main()