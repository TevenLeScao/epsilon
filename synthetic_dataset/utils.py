def fill_in_template(template: str, object1: str, object2: str):
    return template.replace("$1", object1).replace("$2", object2)
