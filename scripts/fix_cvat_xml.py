import sys
import xml.etree.ElementTree as ET

src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
for track in tree.getroot().iter("track"):
    for box in track.iter("box"):
        box.set("outside", box.get("outside", "0"))
        box.set("occluded", box.get("occluded", "0"))
        box.set("keyframe", box.get("keyframe", "1"))
tree.write(dst, encoding="utf-8", xml_declaration=True)
print("fixed:", dst)