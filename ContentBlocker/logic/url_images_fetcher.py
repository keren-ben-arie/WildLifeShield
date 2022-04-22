from bs4 import *
import requests
import io
import PIL.Image as Image


# DOWNLOAD ALL IMAGES FROM THAT URL
def get_actual_images(images):
    # initial count is zero
    count = 0

    # print total images found in URL
    print(f"Total {len(images)} Image Found!")
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
                    r = str(r, 'utf-8')  # TODO convert to png

                except UnicodeDecodeError:
                    # image = pyvips.Image.new_from_file(r, dpi=300)
                    # image.write_to_file("img.png")
                    img = Image.open(io.BytesIO(r))
                    print(type(img))
                    images_array.append(img)

                    # counting number of image downloaded
                    count += 1
            except:
                pass

        # There might be possible, that all
        # images not download
        # if all images download
        if count == len(images):
            print("All Images Downloaded!")

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
