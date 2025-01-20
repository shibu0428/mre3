import socket
import struct
import time

# UDPの受信設定
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#udp_socket.bind(("192.168.10.110", 5002))  # IPアドレスを指定してバインド
udp_socket.bind(("133.83.82.105", 5002))
#udp_socket.bind(("10.20.145.63", 5002))
tag='_haya_'+'1'
print("Waiting for UDP data")
nframe=0
dataframe=600
minframe=300

maxframe=dataframe+minframe

list_motions=[
    'vslash',
    'hslash_ul',
    'hslash_ur',
    'thrust',
    'noutou_koshi',
    'noutou_senaka',
    'roll_r',
    'roll_l',
    'walk',
    'golf',
]

for motion in list_motions:
    print(motion+' set')
    filename=motion+tag
    nframe=0

    with open(filename+'.csv',mode='a') as f:
        while True:
            data, addr = udp_socket.recvfrom(1024)  # データを受信
            nframe+=1
            if nframe==1:
                print("start")
            #開始のフレームより早いなら処理しない
            if nframe<minframe:
                continue
            
            if nframe==minframe:
                start_time = time.time()

            if nframe%100==0:
                print(nframe-minframe)

            #終了フレームを過ぎたらレコーディングを終了してデバック表示&exit
            if nframe>maxframe:
                end_time = time.time()
                exe_time = end_time - start_time
                print("レコーディング完了\n",
                        "ファイル名",filename,".csv\n",
                        "フレーム数",dataframe,"\n",
                        "計測時間",exe_time,"\n",
                        "終了します\n")
                continue
            #バイト列を処理して書き込み
            for i in range(6):
                f.write(str(struct.unpack('<ffffffff', data[i*40:i*40+32])).replace(")(", ",").replace("(", "").replace(")", ""))
                f.write(",")
                f.write(str(struct.unpack('<q', data[i*40+32:i*40+40])).replace(")(", ",").replace("(", "").replace(")", ""))
            f.write(f"\n")
    print(motion+" saved")
    
    print('change motion while 400F')
    for i in range(400):
        data_meta, addr_meta = udp_socket.recvfrom(1024)  # データを受信
        if i %100==0:
            print(i)
    print(next) 