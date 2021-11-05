"""
作成日時:2021/10/28
作成者: Masaaki Sakagami
作成目的： Pythonを使用してDAQ及びVISA製品(今回はFG)を制御する
作業内容：
1. csvから刺激の波形情報を持ってくる(超音波音ともにランダマイズさせるつもり・石坂さんのプログラムを参考にする)
2. inputもするかoutputも込でするかでプログラム切り替える
3. outputする場合は石坂さんのプログラムを参考に操作
4. 超音波刺激の場合には戸田さんのプログラムを参考にして超音波刺激を出力
"""

# ライブラリの読みこみ

import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import datetime as dt
import random
import pandas
import sys
import json
import pickle
import os

import pyvisa as visa
import nidaqmx
from nidaqmx import stream_writers
from nidaqmx import constants,stream_readers

from multiprocessing.pool import ThreadPool
import threading


# キャリブレーションファイルの読み込み
tone_calibrate_filename = "pure_calib_sakagami.csv"
calib_pure = pandas.read_csv(tone_calibrate_filename)

click_calibrate_filename = "click_calib_sakagami.csv"
calib_click = pandas.read_csv(click_calibrate_filename)


def table(**kwargs):
    # 各種刺激のパラメータをキーを行パラメータ、値をセルに入力するように整形
    keys = list(kwargs.keys())
    values = np.array(np.meshgrid(*tuple(kwargs.values()))).transpose()
    return pandas.DataFrame(values.reshape(np.prod(values.shape[:-1]), values.shape[-1]), columns=keys)


def click(sampling_rate, duration, db):
    """
    args:
        sampleing_rate (float) : サンプリング周波数(1秒間に何回の記録点が存在するか)[Hz]
        duration (float): 音刺激の長さ[s]
        db (int): 出力したい音の音圧 [dB]
    """
    amplitude = calib_click[calib_click.db == db].amplitude.values
    if duration<0.002:
        wave = np.zeros(int(sampling_rate*0.0015))
    else:
        wave = np.zeros(int(sampling_rate*duration*2))
        
    wave[0: int(sampling_rate*duration)] = amplitude

    if duration<0.002:
        trigger = np.zeros(int(sampling_rate*0.0015))
    else:
        trigger = np.zeros(int(sampling_rate*duration*2))
    trigger[0: int(sampling_rate*0.001)] = 5

    return np.array([wave, trigger])


def toneburst(sampling_rate, frequency, db, rise_fall, duration):
    """
    args:
        sampling_rate (float) : サンプリング周波数(1秒間に何回の記録点が存在するか)[Hz]
        frequency (float): 生成音の周波数[Hz]
        db (int): 出力したい音の音圧 [dB]
        rise_fall: 立ち上がり/下り時間(片側時間) [s],
        duration: 音刺激の長さ [s]
    """
    fade = int(rise_fall*sampling_rate)
    amplitude = calib_pure[(calib_pure.frequency == frequency) & (
        calib_pure.db == db)].amplitude.values
    wave = amplitude*np.sin(np.linspace(0, np.pi *
                                        2*frequency*duration, int(duration*sampling_rate)))
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    trigger = np.zeros(int(sampling_rate*duration))
    trigger[0: int(sampling_rate*0.001)] = 5

    return np.array([wave, trigger])

# 超音波刺激用のtriggerとenvelopeを"[daq_trigger,fungene_trigger,envelope]"の形で作成

def us_burst(sampling_rate, duration, pulse_duration, window, prf,offset=0.2):
    # 振幅変調モードで増幅するかどうか不明なため取り敢えず、ampを振幅波で入れる
    # もし増幅されてしまうようなら増幅値を抜く
    window_duration = int(sampling_rate*pulse_duration*(window/100))
    pulse_envelop = 1*np.ones(int(pulse_duration*sampling_rate))
    # 窓長がある場合には窓をかける
    # 片側が最大値2/piになるように
    if window != 0:
        pulse_envelop[:window_duration] = np.sin(
            np.pi*(np.arange(window_duration)/(window_duration*2)))
        pulse_envelop[-window_duration:] = np.sin(
            np.pi*(np.arange(window_duration, 0, -1)/(window_duration*2)))
    envelop_wave = np.zeros(int(sampling_rate*duration))
    # prf周期でpulse_envelopを印加
    i = 0
    pulse_interval = int(sampling_rate/prf)
    #ブロードキャストエラー→最後のパルスが格納できなかった説？
    while i*pulse_interval < duration*sampling_rate-len(pulse_envelop):
        envelop_wave[i*pulse_interval:i*pulse_interval+len(pulse_envelop)] = pulse_envelop
        i += 1
    trigger = np.zeros(int(sampling_rate*duration))
    trigger[0: int(sampling_rate*0.001)] = 5
    offset_span=np.zeros(int(offset*sampling_rate))
    envelop_wave=np.concatenate([offset_span,envelop_wave])
    trigger=np.concatenate([offset_span,trigger])
    return np.array([trigger, trigger, envelop_wave])


def us_cont(sampling_rate, duration,window,offset=0.2):
    window_duration = int(sampling_rate*window)
    envelop_wave = np.ones(int(sampling_rate*duration))
    if window != 0:
        envelop_wave[:window_duration] = np.sin(
            0.5*np.pi*(np.arange(window_duration))/window_duration)
        envelop_wave[-window_duration:] = np.sin(
            0.5*np.pi*((window_duration-np.arange(window_duration))/window_duration))
    trigger = np.zeros(int(sampling_rate*duration))
    trigger[0: int(sampling_rate*0.001)] = 5
    offset_span=np.zeros(int(offset*sampling_rate))
    envelop_wave=np.concatenate([offset_span,envelop_wave])
    trigger=np.concatenate([offset_span,trigger])

    return np.array([trigger, trigger, envelop_wave])

# 使用可能なデバイス一覧を取得


def get_devices(resource_manager):
    devise_list = list(resource_manager.list_resources())
    return devise_list

# resource_managerから特定機器を取得


def open_device(resource_manager,address):
    return resource_manager.open_resource(address)

# 機器情報を取得


def identification(device):
    return device.query("*IDN?")

# fungeneの動作確認


def check_fungene_status(index,resource_manager):
    print("checking fungene status \n connecing devices are ...")
    # 接続機器リストを取得
    device_list = get_devices(resource_manager)
    print(device_list)
    # ファンクションジェネレータのデータを取得
    fungene_device = open_device(resource_manager,device_list[index])
    # 取得したdeviceの機器情報を取得
    print("using device informations:")
    print(identification(fungene_device))

# 超音波用刺激設定


def fungene_stim_setting(fungene_device, amp):
    # py_visaを使って超音波波形生成→基本連続波にしてDAQからの振幅変調波で波形調整
    #fungene_device.write(":SOURce1:AMSC:STATe ON ")
    fungene_device.write(":SOURce1:AMSC:SOURce EXTernal")
    fungene_device.write(":SOURce1:FUNCtion:SHAPe SIN")
    fungene_device.write(":SOURce1:FREQuency:CW 500000HZ")
    if amp > 6.0:
        print("最大値を超過しています")
        sys.exit()
    fungene_device.write(
        f":SOURce1:VOLTage:LEVel:IMMediate:AMPLitude {amp}VPP")
    fungene_device.write(":SOURce1:AMSC:DEPTh 100.0PCT")
    fungene_device.write(":SOURce1:PHASe:ADJust 0DEG")
    fungene_device.write(":SOURce1:AMSC:STATe OFF")
    fungene_device.write(":OUTPut:STATe 0")
    #fungene_device.write(":OUTPut1:POLarity SINusoid, NORMal")
    #fungene_device.write(":OUTPut1:SCALe SINusoid, FS")
    
def fungene_on(fungene_device):
    #外部変調オン
    fungene_device.write(":SOURce1:AMSC:STATe ON")
    fungene_device.write(":OUTPut:STATe 1")
    


def fungene_off(fungene_device):
    # outputをoffに設定
    fungene_device.write(":OUTPut:STATe 0")
    fungene_device.write(":SOURce1:AMSC:STATe OFF")
    
    
    
    
    

def input_data(sampling_rate, input_duration):
    #device=nidaqmx.system.device.Device("Dev1")
    #device.reset_device()
    task = nidaqmx.Task()
    print("read: "+task.name)
    task.ai_channels.add_ai_voltage_chan('Dev1/ai0:1')
    analog_reader=stream_readers.AnalogMultiChannelReader(task.in_stream)
    task.timing.cfg_samp_clk_timing(
        rate=sampling_rate, samps_per_chan=int(sampling_rate*(input_duration)))
    one_channel_data=np.zeros(int(sampling_rate*(input_duration)))
    data=np.array([one_channel_data,one_channel_data])
    task.start()
    analog_reader.read_many_sample(data=data,number_of_samples_per_channel=int(sampling_rate*input_duration))
    task.wait_until_done()
    task.stop()
    task.close()
    return data



def output_stimulation(data, sampling_rate, state,fungene_device=None,amp=0,is_imput=False):
    ## 超音波用にFGの設定 ##
    if state == "us_burst" or state == "us_cont":
        fungene_stim_setting(fungene_device,amp)
    #######################
    if not is_imput:
        device=nidaqmx.system.device.Device("Dev1")
        device.reset_device()
    task = nidaqmx.Task()
    # 出力用チャンネルの生成
    if state == "click" or state == "toneburst":
        task.ao_channels.add_ao_voltage_chan('Dev1/ao0:1')
    elif state == "us_burst" or state == "us_cont":
        task.ao_channels.add_ao_voltage_chan('Dev1/ao1:3')
    else:
        print("適切な刺激を設定してください")
        return 0
    # バッファ上に波形を書き込む際に書き込み用のクラスをtaskのout_streamと結び付ける
    analog_writer = stream_writers.AnalogMultiChannelWriter(task.out_stream)
    # analog_writerを使用して書き込みを行う
    if state == "click" or state == "toneburst":
        task.timing.cfg_samp_clk_timing(rate=sampling_rate, samps_per_chan=data.size//2)
    elif state == "us_burst" or state == "us_cont":
        task.timing.cfg_samp_clk_timing(rate=sampling_rate, samps_per_chan=data.size//3)
    # writeする際にバッファサイズが足りないというエラーが出た→バッファサイズ指定の関数が必要なのかな？→単に前のtaskが残っていただけ
    analog_writer.write_many_sample(data)
    
    print("write: "+task.name)
    task.start()
    if state == "us_burst" or state == "us_cont":
        fungene_on(fungene_device)
    # すぐに止めると電圧値が変な値で止まるため待機させる
    task.wait_until_done()
    task.stop()
    # タスクの終了の間にタイムラグがある可能性があるためタスクの終了前にFGのoutputをオフにする
    if state == "us_burst" or state == "us_cont":
        fungene_off(fungene_device)
    task.close()
    


def output_and_input(data, input_duration, sampling_rate, state,fungene_device=None,amp=0):
    # Dev1の初期化これ意外と大事
    device = nidaqmx.system.device.Device("Dev1")
    device.reset_device()
    # 複数処理用のコード（並列処理を行う)
    pool = ThreadPool(processes=1)
    result = pool.apply_async(input_data,(sampling_rate, input_duration))
    time.sleep(0.05) #大事な処理
    worker = threading.Thread(
        target=output_stimulation,args=(data, sampling_rate, state,fungene_device,amp,True))
    worker.start()

    worker.join()
    record = result.get()
    pool.close()
    return record


def stim_trial(data, is_read, input_duration, sampling_rate, state, fungene_device=None,amp=0):
    if is_read:
        result = output_and_input(data, input_duration, sampling_rate,state,fungene_device)
    else:
        result = output_stimulation(data, sampling_rate, state,fungene_device,amp)
    return result


def stim_protocol(is_flavo=False,resource_manager=None):
    # 刺激パラメータの設定や生成を行う関数
    # 基本いじるのはこことoutput_paradigmのintervalだけでOK

    ########################
    ## 刺激パラメータの設定 ##
    ########################

    d_today = dt.datetime.today()
    filename = d_today.strftime("%y%m%d_%H%M")+"_order.csv"

    # trial数(個別に回数変更したい場合にのみ選択)
    trial = 3  # [times]
    if is_flavo:
        trial = 10

    #clickの設定###
    click_duration = 0.0001  # [s]
    click_db = np.arange(30, 90, 10)
    #click_db = np.array([80])
    if is_flavo:
        click_db = np.array([60])
    click_trial = trial

    # click用のtrial設定
    click_order = table(db=click_db, duration=click_duration,
                        trial=range(click_trial))
    click_order["state"] = "click"

    # toneburstの設定
    pure_freq = calib_pure.frequency.unique()
    pure_db = np.arange(30, 90, 10)
    if is_flavo:
        pure_freq = np.array([2000,4000,8000,16000])
        pure_db = np.array([60])
    #pure_db = np.array([80])
    pure_duration = 0.05  # [s]
    pure_rise_fall = pure_duration*0.1  # [s]
    pure_trial = trial  # 各周波数の呈示回数

    # toneburst用のtrial設定
    pure_order = table(db=pure_db, frequency=pure_freq, duration=pure_duration,
                       rise_fall=pure_rise_fall, trial=range(pure_trial))
    pure_order["state"] = "toneburst"

    # ultrasound_burst_trainの設定
    us_burst_amplitude = np.array([3.0])  # [Vpp]
    us_burst_duration = 0.03  # [s]
    us_burst_pulse_duration = 0.00016  # [s]
    us_burst_window_percentage = np.array([20])  # [%]
    us_burst_PRF = 1500  # [Hz]
    us_burst_trial = trial

    # us_burst用のtrial設定
    us_burst_order = table(amp=us_burst_amplitude, duration=us_burst_duration,
                           pulse_duration=us_burst_pulse_duration, window=us_burst_window_percentage,
                           PRF=us_burst_PRF, trial=range(us_burst_trial))
    us_burst_order["state"] = "us_burst"

    # ultrasound_continuousの設定
    us_cont_amplitude = np.array([3.0])
    us_cont_duration = np.array([0.03])
    us_cont_window_duration = np.array([0])  # [s]
    us_cont_trial = trial

    # us_continuous用のtrial設定
    us_cont_order = table(amp=us_cont_amplitude, duration=us_cont_duration,
                          window=us_cont_window_duration, trial=range(us_cont_trial))
    us_cont_order["state"] = "us_cont"

    ###音刺激の順番決定###
    #use_stims = [us_cont_order,us_burst_order,click_order]
    use_stims=[us_burst_order]
    if is_flavo:
        use_stims=[pure_order]
    #dataframe_objectはここでエラーを起こすみたい→データの行列が一致していないと比較不可能
    
    order = pandas.concat(use_stims, axis=0).sample(
        frac=1).reset_index(drop=True)
    if (order == "us_burst").values.sum() + (order == "us_cont").values.sum() > 0:
        check_fungene_status(0,resource_manager)
    if is_flavo:
        # フラビンの場合は順番だけランダマイズ
        order = order.drop("trial", axis=1).drop_duplicates()
        order = pandas.concat([order for _ in range(trial)],
                              axis=0).reset_index(drop=True)
    order.to_csv("order/" + filename, index=False)

    return order


def stim_paradigm(is_read=False, is_flavo=False):
    """
    刺激生成用のプログラム
    超音波と音刺激のランダマイズを行えるようにしたい
    """
    ########################
    ## 時間パラメータの設定 ##
    ########################

    # 計測パラメータの入力
    # 1MHzのサンプリングレートだとエラーが出る
    sampling_rate = 4*10**5

    ## intervalの設定 ##
    interval = 2  # [s]
    waggle = 0.05  # [s]
    if is_flavo:
        interval = 8.0
        waggle = 0

    # input用のパラメータ
    input_duration = 2

    # fungene制御用の変数
    resource_manager = visa.ResourceManager()
    device_list = get_devices(resource_manager)
    fungene_device = open_device(resource_manager,device_list[0])
    order = stim_protocol(is_flavo,resource_manager)

    

    if is_read:
        result_dic = {}

    # DAQに書き込む出力波形データの選択
    for i, row in order.iterrows():
        print("number : {0} / {1}\n{2}".format(i+1, order.shape[0], row))

        if row.state == "click":
            data = click(sampling_rate, row.duration, row.db)
            # nidaqmxライブラリを使用した音波形生成
        elif row.state == "toneburst":
            data = toneburst(sampling_rate, row.frequency,
                             row.db, row.rise_fall, row.duration)
        elif row.state == "us_burst":
            data = us_burst(sampling_rate, row.duration,
                            row.pulse_duration, row.window, row.PRF)
        elif row.state == "us_cont":
            data = us_cont(sampling_rate, row.duration, row.window)
        else:
            print("音刺激がありません")

        ##daqのコントロール##
        # 波形の生成及び計測
        if is_read:
            result = stim_trial(data, is_read, input_duration,
                                sampling_rate, row.state,fungene_device,row.amp)

            # 結果配列の格納方法
            if row.state not in result_dic:
                result_dic[row.state] = {}

            if row.state == "click":
                if row.db not in result_dic[row.state]:
                    result_dic[row.state][row.db] = []
                result_dic[row.state][row.db].append(result)

            elif row.state == "tone_burst":
                if row.freq not in result_dic[row.state]:
                    result_dic[row.state][row.freq] = {}
                if row.db not in result_dic[row.state][row.freq]:
                    result_dic[row.state][row.freq][row.db] = []
                result_dic[row.state][row.freq][row.db].append(result)

            elif row.state == "us_burst":
                if row.duration not in result_dic[row.state]:
                    result_dic[row.state][row.duration] = {}
                if row.pulse_duration not in result_dic[row.state][row.duration]:
                    result_dic[row.state][row.pulse_duration] = {}
                if row.PRF not in result_dic[row.state][row.duration]:
                    result_dic[row.state][row.pulse_duration][row.PRF] = {}
                if row.amp not in result_dic[row.state][row.duration][row.PRF]:
                    result_dic[row.state][row.pulse_duration][row.PRF][row.amp] = {}
                if row.window not in result_dic[row.state][row.pulse_duration][row.PRF][row.amp]:
                    result_dic[row.state][row.pulse_duration][row.PRF][row.amp][row.window] = [
                    ]
                result_dic[row.state][row.pulse_duration][row.PRF][row.amp][row.window].append(
                    result)

            elif row.state == "us_cont":
                if row.duration not in result_dic[row.state]:
                    result_dic[row.state][row.duration] = {}
                if row.amp not in result_dic[row.state][row.duration]:
                    result_dic[row.state][row.duration][row.amp] = {}
                if row.window not in result_dic[row.state][row.duration][row.amp]:
                    result_dic[row.state][row.duration][row.amp][row.window] = []
                result_dic[row.state][row.duration][row.amp][row.window].append(
                    result)
        else:
            # 波形の出力
            if row.state=="us_burst" or row.state=="us_cont":
                stim_trial(data, is_read, input_duration, sampling_rate, row.state,fungene_device,row.amp)
            else:
                stim_trial(data, is_read, input_duration, sampling_rate, row.state,fungene_device,)

        # interstimulus interval[s]
        isi = interval + random.uniform(-1, 1)*waggle
        print("waittime : ", isi)
        print("\n")
        time.sleep(isi)
    if is_read:
        # result_dicをjson方式で出力
        today_file =dt.datetime.now().strftime("%y%m%d_%H%M")
        with open(f"data/{today_file}-data.json", 'w') as f:
            json.dump(result_dic, f, indent=2)
        # print(result_dic)


def main():
    # キャリブレーションファイルの読み込み

    calib_pure = pandas.read_csv(tone_calibrate_filename)
    calib_click = pandas.read_csv(click_calibrate_filename)
    
    print(f"tone_csv_file:\n {tone_calibrate_filename}")
    print(f"click_csv_file:\n {click_calibrate_filename}")

    # 計測状況の選択
    print("DAQでの電圧計測も行いますか？ (please type \"y\" / \"n\") ")
    record_command = input()

    if record_command == "y":
        is_read = True
        # 結果保存用のディレクトリを生成
        if not(os.path.exists("./data")):
            os.mkdir("./data")
    else:
        is_read = False

    print("フラビン計測ですか？ (please type \"y\" / \"n\") ")
    record_command = input()

    if record_command == "y":
        is_flavo = True
    else:
        is_flavo = False

    if not(os.path.exists("./order")):
        os.mkdir("./order")
    stim_paradigm(is_read, is_flavo)

    # 刺激の順番決定
    today = dt.datetime.now()
    print(f"END : {today}")


if __name__ == "__main__":
    main()
