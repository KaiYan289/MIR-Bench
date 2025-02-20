import re
import torch
import sys
import json
import random, string
from typing import List, Tuple
import math
import bisect
from bisect import *
from collections import *
import heapq

res = []
def numerical_letter_grade(grades):
    """It is the last week of the semester and the teacher has to give the grades
    to students. The teacher has been making her own algorithm for grading.
    The only problem is, she has lost the code she used for grading.
    She has given you a list of GPAs for some students and you have to write 
    a function that can output a list of letter grades using the following table:
             GPA       |    Letter grade
              4.0                A+
            > 3.7                A 
            > 3.3                A- 
            > 3.0                B+
            > 2.7                B 
            > 2.3                B-
            > 2.0                C+
            > 1.7                C
            > 1.3                C-
            > 1.0                D+ 
            > 0.7                D 
            > 0.0                D-
              0.0                E
    

    Example:
    grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']
    """


    def to_letter_grade(score):
      if score == 4.0:
        return "A+"
      elif score > 3.7:
        return "A"
      elif score > 3.3:
        return "A-"
      elif score > 3.0:
        return "B+"
      elif score > 2.7:
        return "B"
      elif score > 2.3:
        return "B-"
      elif score > 2.0:
        return "C+"
      elif score > 1.7:
        return "C"
      elif score > 1.3:
        return "C-"
      elif score > 1.0:
        return "D+"
      elif score > 0.7:
        return "D"
      elif score > 0.0:
        return "D-"
      else:
        return "E"
    
    return [to_letter_grade(x) for x in grades]


res.append(numerical_letter_grade(*[[2.7, 1.7, 3.0, 3.7, 0.0, 0.7]]))
res.append(numerical_letter_grade(*[[3.7, -0.0, 3.0, 1.7, 0.0, 1.7, 4.0, 3.0, 2.3, 0.7, 0.7, 2.7, 0.7, 4.0]]))
res.append(numerical_letter_grade(*[[3.7, 0.7, 1.0, 3.7, 1.3, 4.0, 2.7, 3.3, 1.7, 4.0, 4.0, 4.0, 1.0, 2.7, 3.0]]))
res.append(numerical_letter_grade(*[[3.0, 1.0, 1.3, -0.0, 0.7, 1.7, 1.3, 1.7, 2.7, 2.0, 1.0]]))
res.append(numerical_letter_grade(*[[4.0, 0.7, -0.0, -0.0, 2.7, 2.7, 1.7, 4.0, 1.3]]))
res.append(numerical_letter_grade(*[[1.3, 2.3, 4.0, 2.3, 1.0, 2.7, 1.3, 4.0, 0.7]]))
res.append(numerical_letter_grade(*[[-0.0, -0.0, 1.7, 4.0, 1.0, 1.7, 2.7, 2.3, 2.0, 0.7, 0.7, 0.7, 2.3, 1.7]]))
res.append(numerical_letter_grade(*[[2.3, 0.0, 2.0, 4.0, 0.0, 3.0, 1.0, 2.0, 1.7]]))
res.append(numerical_letter_grade(*[[1.3, 4.0, 2.3, 3.3, 2.0, 2.3]]))
res.append(numerical_letter_grade(*[[2.7, 1.0, 1.7, 3.7, 1.0, 3.7, 2.7, 2.0, 3.0, 2.3, 1.7, 2.0, -0.0, 3.0]]))
torch.save(res, '../data_exec_output/81_testset_data.pt')