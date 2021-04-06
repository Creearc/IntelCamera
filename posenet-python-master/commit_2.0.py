# -*- coding: cp1251 -*-
from ftplib import FTP
import sys
fl = []
if len(sys.argv) > 1:
  for i in range(1, len(sys.argv)):
    fl.append(sys.argv[i].split('\\')[-1])
else:
  fl.append('stream.py')


for j in range(len(fl)):
  print(fl[j])

  ftp = FTP()
  HOSTS = ['192.168.68.119']
  PORT = 21
  for i in range(len(HOSTS)):
    ftp.connect(HOSTS[i], PORT)
    print(ftp.login(user='pi', passwd='9'))

    ftp.cwd('poses')

    with open(fl[j], 'rb') as f:
        ftp.storbinary('STOR ' + fl[j], f, 1024)

    print('Done!')


