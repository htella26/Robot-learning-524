from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class BottleObject(MujocoXMLObject):
    """
    Bottle object that loads a custom mesh-based bottle from XML.

    The XML should define mesh, material, and collision geoms. Sites for top,
    bottom, and horizontal radius can be used for observation or control.
    """

    def __init__(self, name="bottle"):
        # This loads robosuite/models/assets/objects/bottle.xml
        super().__init__(
            xml_path_completion("objects/hoh_bottle.xml"),
            name=name,
        )
