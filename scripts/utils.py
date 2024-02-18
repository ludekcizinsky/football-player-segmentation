def get_image_annotations(ann : list, img_id : int) -> list:
    selected = []
    for a in ann:
        if a['image_id'] == img_id:
            selected.append(a)
    return selected