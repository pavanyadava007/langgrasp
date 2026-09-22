from langgrasp.language.parser import parse_command


def test_plain():
    i = parse_command("pick the blue screwdriver")
    assert i.action == "pick" and i.color == "blue" and i.noun == "screwdriver" and i.phrase == "blue screwdriver"
    assert i.spatial is None and not i.generic


def test_synonym_and_politeness():
    i = parse_command("Can you grab the yellow driver, please?")
    assert i.action == "pick" and i.color == "yellow" and i.noun == "driver" and i.phrase == "yellow driver"


def test_attribute_only():
    i = parse_command("grab the blue one")
    assert i.generic and i.noun is None and i.color == "blue" and i.phrase == "blue object"


def test_spatial():
    i = parse_command("pick the red cube on the left")
    assert i.spatial == "left" and i.phrase == "red cube"
    i = parse_command("pick up the green can on the right")
    assert i.spatial == "right" and i.noun == "can" and i.action == "pick"


def test_destination_stripped():
    i = parse_command("put the red cube in the tray")
    assert i.action == "place" and i.phrase == "red cube"


def test_unknown_object_passthrough():
    i = parse_command("pick the stapler")
    assert i.noun == "stapler" and i.phrase == "stapler" and i.color is None
