from jina import Flow, Document, Client
import numpy as np
from PIL import Image


class UI:
    about_block = """
    """
    css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: "#111";
        background-color: "#eee";
    }}
</style>
"""


headers = {"Content-Type": "application/json"}


def get_breed(query, host='0.0.0.0', protocol='grpc', port=12345):
    client = Client(host=host, protocol=protocol,
                    port=port, return_responses=True)
    image = Image.open(query)
    img_array = np.array(image, dtype=np.uint8)
    doc = Document(tensor=img_array)
    resp = client.post(on='/', inputs=doc)
    return resp[0].docs[0].tags, image
