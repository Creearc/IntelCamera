# -*- coding: cp1251 -*-
from ftplib import FTP

ftp = FTP()
#HOST = '165.22.84.248'
HOST = '192.168.68.119'
PORT = 21
ftp.connect(HOST, PORT)
#print(ftp.login(user='andrew', passwd='fshhrt'))
#print(ftp.login(user='pi', passwd='9'))
print(ftp.login(user='pi', passwd='9'))

ftp.cwd('intel/out/')
#ftp.cwd('/media/linaro/Transcend/Video/')
#ftp.cwd('0')
#data = ftp.retrlines('LIST')
#print(data)

fl = '1611267106.1079283.png'
out = 'out/' + fl

with open(out, 'wb') as f:
    ftp.retrbinary('RETR ' + fl, f.write)

