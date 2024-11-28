import time
print("aaa")
for i in range(10):
    print('\rNo, %d' % i, end='')
    time.sleep(0.5)