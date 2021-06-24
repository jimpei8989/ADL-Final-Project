import random


def draw_from_list(a: list):
    return a[int(random.random() * len(a))]
