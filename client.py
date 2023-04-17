import base64

import jsonpickle
import numpy as np
import requests
from matplotlib import pyplot as plt

path = '/media/banglv/4TbDATA/backup_banglv1/django_removal/data/ex1.jpg'
payload = {"image": open(path, "rb")}
url = "http://127.0.0.1:8000/execute/bgremoval/"
predict_params = {'style': 'artstyle'}  #
r = requests.post(url,
                  files=payload,
                  data={'predict_params': jsonpickle.encode(predict_params)}).json()
if r['success']:
    b64string = r['result_image']
    f = base64.b64decode(b64string.encode("utf-8"))
    one_dim = np.frombuffer(f, dtype=np.uint8)
    predicted = one_dim.reshape(r['w'], r['h'])
    plt.imshow(predicted, cmap='gray')
    plt.show()