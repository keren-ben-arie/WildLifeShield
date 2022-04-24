import pickle

import numpy as np
import torch
from PIL import Image
from django.contrib.auth.mixins import PermissionRequiredMixin
from django.contrib.messages import INFO
from django.views.generic import TemplateView
from requests.exceptions import MissingSchema
from torchvision import transforms
import torch.nn.functional as F

from ContentBlocker.logic.url_images_fetcher import get_images_from_url
from django.shortcuts import redirect
from django.contrib import messages

from ModelCreation.CNN import CNN


class OnlineSamplerView(PermissionRequiredMixin, TemplateView):
    template_name = "index.html"
    permission_required = []
    ready_cnn = CNN()
    ready_cnn.load_state_dict(torch.load("C:\\Users\\User\\PycharmProjects\\CageClassifier\\ModelCreation"
                                         "\\trained_model.pt"))
    ready_cnn.eval()
    models = {"caged-animals": ready_cnn}

    @classmethod
    def as_view(cls, **initkwargs):
        view = super().as_view(**initkwargs)
        return view

    def get_title(self):
        return "WildLifeShield"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["message"] = self._get_message()
        context["classifier"] = self.models.keys()
        return context

    def post(self, request, *args, **kwargs):
        classifier = request.POST.get('classifier')
        print(classifier)
        try:
            url = request.POST.get('url')
            images = get_images_from_url(url)
        except MissingSchema:
            messages.error(self.request, f"The URL is invalid.")
            return redirect(self.request.path)
        cages_counter = 0
        ready_images = []
        for i, img in enumerate(images):
            self.models[classifier].eval()
            img = img.convert('RGB')
            transform_image = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            ready_image = transform_image(img).float()
            ready_image = ready_image.unsqueeze(0)
            ready_images.append(ready_image)
            y = self.models[classifier](ready_image)
            probs = torch.nn.functional.softmax(y, dim=1)
            conf, predicted = torch.max(probs, 1)
            if predicted[0] == 1:
                if conf < 0.7:
                    predicted[0] = 0
            cages_counter += predicted[0]
            print(i , " ",predicted," ", conf)

        if cages_counter > 0:
            messages.error(self.request, f"There are {len(images)} on this webpage, out of them: {cages_counter} are unsafe for you.")
        else:
            messages.success(self.request, f"This webpage is safe. There are {len(images)} which are completely safe.")
        # certainty = ready_cnn.score(X_test, Y_test)
        return redirect(self.request.path)

    def _get_message(self):
        storage = getattr(self.request, '_messages', [])
        if not storage:
            return None
        message = [message for message in storage][-1].message
        return message

    def set_message(self, message, level=INFO):
        messages = self.request._messages
        messages.add(level, message)
