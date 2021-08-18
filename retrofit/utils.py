# Module: utils
# Author: Adrian Antico <adrianantico@gmail.com>
# License: Mozilla Public License 2.0
# Release: retrofit 0.0.1
# Last modified : 2021-08-17

def ClearMemory():
  """
  Remove all files, variables, objects, etc. from memory. Total wipe
  """
  for element in dir():
    if element[0:2] != "__":
      del globals()[element]
