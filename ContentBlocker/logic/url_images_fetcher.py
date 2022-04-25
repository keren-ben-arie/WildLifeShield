from bs4 import *
import requests
import io
import PIL.Image as Image
# Imports the Google Cloud client library
from google.cloud import vision

client = vision.ImageAnnotatorClient()

def is_animal(content):
    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = [label.description.split() for label in response.label_annotations]
    flat_labels = [item for sublist in labels for item in sublist]
    print(flat_labels)
    return 'animal' in flat_labels or 'Animal' in flat_labels or 'Mammal' in flat_labels


# DOWNLOAD ALL IMAGES FROM THAT URL
def get_actual_images(images):
    # initial count is zero
    count = 0

    # print total images found in URL
    images_array = []

    if len(images) != 0:
        for i, image in enumerate(images):
            try:
                image_link = image["data-srcset"]
            except:
                try:
                    image_link = image["data-src"]
                except:
                    try:
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            image_link = image["src"]
                        except:
                            pass
            # After getting Image Source URL
            # We will try to get the content of image
            try:
                r = requests.get(image_link).content
                try:
                    # possibility of decode
                    r = str(r, 'utf-8')

                except UnicodeDecodeError:
                    if not is_animal(r):
                        continue
                    img = Image.open(io.BytesIO(r))
                    width, height = img.size
                    if width < 100 or height < 100:
                        continue
                    images_array.append(img)
                    count += 1
                    print(i, image_link)
            except:
                pass

    print(f"Total {len(images_array)} Image Found!")
    return images_array


# MAIN FUNCTION START
def get_images_from_url(url):
    # content of URL
    r = requests.get(url)

    # Parse HTML Code
    soup = BeautifulSoup(r.text, 'html.parser')

    # find all images in URL
    images = soup.findAll('img')

    net_input = get_actual_images(images)

    return net_input
