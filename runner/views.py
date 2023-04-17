import base64

# import jsonpickle
import numpy as np
from django.http import JsonResponse
# from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from runner.src.processor import BackgroundRemoval
from utils.utils import _grab_image

processor = BackgroundRemoval('runner/src/saved_models/IS-Net/isnet-general-use.pth')

@csrf_exempt
def background_removal_apps(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False, 'result_image': 'AI core failed'}
    # check to see if this is a post request
    if request.method == "POST":
        # predict_params = jsonpickle.decode(request.POST.get("predict_params", None))
        # style = predict_params['style']
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image = _grab_image(url=url)

        try:
            result = processor.image_prediction(image)
            b64string = base64.b64encode(result.astype(np.uint8)).decode("utf-8")
            data = {"result_image": b64string,
                    "success": True,
                    'w': result.shape[0],
                    'h': result.shape[1]}
        except:
            data["result_image"] = 'AI core failed'
            data["success"] = False

    return JsonResponse(data=data)