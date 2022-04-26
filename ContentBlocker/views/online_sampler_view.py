import torch
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
    transform_image = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

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
        try:
            url = request.POST.get('url')
            images = get_images_from_url(url)
        except MissingSchema:
            messages.error(self.request, f"The URL is invalid.")
            return redirect(self.request.path)
        cages_counter = 0
        for i, img in enumerate(images):
            img = img.convert('RGB')
            ready_image = self.transform_image(img).float()
            ready_image = ready_image.unsqueeze(0)
            y = self.models[classifier](ready_image)
            probs = torch.nn.functional.softmax(y, dim=1)
            conf, predicted = torch.max(probs, 1)
            cages_counter += predicted[0]
            print(f"Image number {i}, Classification is: {predicted[0]} \n Confidence level is {conf}")

        if cages_counter > 0:
            messages.error(self.request, f"There are {cages_counter} images on this webpage which are unsafe for you.")
        else:
            messages.success(self.request, f"This webpage is safe.")
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
