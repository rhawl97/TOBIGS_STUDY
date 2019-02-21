class Ban:
    def __init__(self, no=None):
        self.no = no
        self.student_list = []

    def __str__(self):
        return "<{}반>".format(self.no)

    def __lt__(self, other):
        return int(self.no) < int(other.no) #오름차순 정렬을 위해

    def __eq__(self, other):
        return int(self.no) == int(other.no)

    def count_student(self):
        return str(len(self.student_list)) + "명"
