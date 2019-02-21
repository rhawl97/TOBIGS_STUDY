import sys  #한글 깨짐 방지
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

text = open('C:/Users/Kim Yuum/Desktop/투빅스/Algorithm/Week4/student.txt', mode='rt', encoding='utf-8') #텍스트 불러오깅
from Student import Student #불러와버리겠다
from Ban import Ban
import numpy as np

num_lines = sum(1 for line in open('C:/Users/Kim Yuum/Desktop/투빅스/Algorithm/Week4/student.txt', mode='rt', encoding='utf-8'))  #학생 수
line = [0]*num_lines #영벡터 생성
info = [0]*num_lines
student = [0]*num_lines
ban_num = [0]*num_lines
ban = []
Ban_list = [0]*5

for lines in range(num_lines):  #한줄씩 읽어들일거야
    line[lines] = text.read().splitlines()  #\n제거하고 읽어오기 #['홍길동 4 1']
    info[lines] = line[0][lines].split(' ') #['홍길동','4','1']
    ban_num[lines] = info[lines][1]  #'4' #몇반인지!
    student[lines] = Student(info[lines][2], info[lines][0]) #Student instance 생성

    if ban_num[lines] in ban:  #이미 존재하는 반이라면
        banname = Ban_list[int(ban_num[lines])-1]  #해당 인덱스에 위치한 반 클래스에 들어가도록!
        banname.student_list.append(student[lines])  #읽어들인 줄 학생 정보 추가
        banname.student_list = sorted(banname.student_list)  #오름차순 정렬
    else:  #아직 추가되지 않은 반이라면
        ban.append(ban_num[lines])  #이미 있는 반인지 확인하기 위해 for문에 사용한 ban!
        banname = Ban(ban_num[lines])  #Ban instance 생성
        Ban_list[int(ban_num[lines])-1] = banname  #Ban list 중 해당 반 위치에 맞게 Ban instance 할당
        banname.student_list.append(student[lines])  #읽어들인 줄 학생 정보 추가
        banname.student_list = sorted(banname.student_list)  #오름차순 정렬

for banname in Ban_list:  #출력
    print("\n", banname, banname.count_student())
    for students in banname.student_list:
        print(students)
