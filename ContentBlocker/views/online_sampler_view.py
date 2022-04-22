import pickle
from django.contrib.auth.mixins import PermissionRequiredMixin
from django.contrib.messages import INFO
from django.views.generic import TemplateView
from requests.exceptions import MissingSchema

from ContentBlocker.logic.url_images_fetcher import get_images_from_url
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages

class OnlineSamplerView(PermissionRequiredMixin, TemplateView):
    template_name = "index.html"
    permission_required = []
    # filename = 'trained_cnn_cage_classifier.sav'
    # ready_cnn = pickle.load(open(filename, 'rb'))
    ready_cnn = None
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
        # y = self.models[classifier].predict(images)
        # for entry in y:
        #     cages_counter += entry
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
