from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class BasketObject(MujocoXMLObject):
    def __init__(self, name="basket"):
        super().__init__(
            xml_path_completion("objects/hoh_basket.xml"),
            name=name,
        )
