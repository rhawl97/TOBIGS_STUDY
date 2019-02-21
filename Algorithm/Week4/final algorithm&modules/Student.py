class Student:
    def __init__(self, id=None, name=None):
        self.id = id
        self.name = name

    def __str__(self):
        return "{}번 {}".format(self.id, self.name)

    def __lt__(self, other):  #오름차순 정렬을 위해
        return int(self.id) < int(other.id)

    def __eq__(self, other):
        return int(self.id) == int(other.id)
