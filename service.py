# coding:utf-8
import argparse
import glob
import json
import math
import librosa
import requests
import time
from flask import Flask, request, jsonify
from pydub import AudioSegment
from tensorpack import SaverRestore, PredictConfig, ChainInit, OfflinePredictor
from werkzeug.utils import secure_filename
import os
import numpy as np
import soundfile as sf
import tensorflow as tf

from models.data_load import normalize_0_1
from models.models import Net2
from hparams.hparam import hparam as hp
from utils.audio import denormalize_db, spec2wav, db2amp, inv_preemphasis, preemphasis, amp2db

app = Flask(__name__, static_url_path='')
predictor = None
logdir2 = None
ckpt2 = None


def init(args=None, is_running=0, pt=None):
    global ckpt2
    # 网络
    model = Net2()
    if is_running == 1:
        if pt == "":
            ckpt2 = tf.train.latest_checkpoint(logdir2)
        else:
            ckpt2 = '{}/{}'.format(logdir2, pt)
    else:
        ckpt2 = '{}/{}'.format(logdir2, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    pred_conf = PredictConfig(
        model=model,
        input_names=['x_ppgs', 'x_mfccs', 'y_spec', 'y_mel'],
        output_names=['pred_spec', "ppgs"],
        session_init=ChainInit(session_inits))
    global predictor
    predictor = OfflinePredictor(pred_conf)
    if is_running == 1:
        return jsonify({"code": 0, "ckpt": ckpt2})


def convert(wav, ppgs):
    """
        转换
        输入:wav,ppgs(将被调整为3秒)
        输出:aduio(3秒)
    """

    # fix wav length
    wav = librosa.util.fix_length(wav, hp.default.sr * hp.default.duration)

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=hp.default.preemphasis)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=hp.default.n_fft, hop_length=hp.default.hop_length,
                     win_length=hp.default.win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    # fix ppgs length
    ppgs = librosa.util.fix_length(ppgs, ((hp.default.duration * hp.default.sr) // hp.default.hop_length + 1))

    # get_input
    x_ppgs, x_mfccs, y_spec, y_mel = (np.arange(9999) == ppgs[:, None]).astype(np.int32), \
                                     mfccs.T, mag_db.T, mel_db.T  # (t,9999)(t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
    x_ppgs, x_mfccs, y_spec, y_mel = x_ppgs[np.newaxis, :], x_mfccs[np.newaxis, :], \
                                     y_spec[np.newaxis, :], y_mel[np.newaxis, :]

    # get_output
    pred_spec, _ = predictor(x_ppgs, x_mfccs, y_spec, y_mel)

    # Denormalization
    pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)

    # db to amp
    pred_spec = db2amp(pred_spec)

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)

    # Spectrogram to waveform
    audio = np.array(
        list(map(lambda spec: spec2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length,
                                       hp.default.n_iter), pred_spec)))

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)

    return audio


def do_service(wav_file):
    # 提取路径
    basepath, filename = os.path.split(wav_file)
    filename, _ = os.path.splitext(filename)

    # 调整采样率和格式
    wav, _ = librosa.load(wav_file, mono=True, sr=hp.default.sr)
    wav_len = np.size(wav)
    sf.write(os.path.join(basepath, filename + ".wav"), wav, hp.default.sr, format="wav", subtype="PCM_16")

    # 读取原始分贝
    sound = AudioSegment.from_file(wav_file, "wav")
    target_dBFS = sound.dBFS

    # 获取ppgs
    multipart_form_data = {
        'wave': ('wav.wav', open(wav_file, "rb"))
    }
    try:
        response = requests.post('http://202.207.12.156:9000/asr', {'ali': 'true'}, files=multipart_form_data)
        content = json.loads(response.text)
        ppgs = np.array(json.loads(content['ali']))
        txt = content['txt']
    except:
        return jsonify({"code": 121, "message": "asr接口请求失败"})

    # 拼接结果
    times = int(math.ceil(wav_len / (hp.default.duration * hp.default.sr)))
    audio = []
    for i in range(times):
        _audio = convert(wav[i * hp.default.duration * hp.default.sr:],
                         ppgs[i * ((hp.default.duration * hp.default.sr) // hp.default.hop_length + 1):])
        audio = audio + _audio[0].tolist()  # _audio[0]

    # times = int(math.ceil(wav_len / (2 * hp.default.sr)))
    # audio = []
    # for i in range(times):
    #     _audio = convert(wav[i * 2 * hp.default.sr:],
    #                      ppgs[i * ((2 * hp.default.sr) // hp.default.hop_length + 1):])
    #     audio = audio + _audio[0].tolist()[:2 * hp.default.sr]  # _audio[0]

    # 修复长度
    audio = librosa.util.fix_length(np.array(audio), wav_len)

    # 写结果
    target_path = os.path.join(os.path.dirname(__file__), "static/uploads", filename + "_output.wav")
    sf.write(target_path, audio, hp.default.sr, format="wav", subtype="PCM_16")

    # 调整分贝
    sound = AudioSegment.from_file(target_path, "wav")
    change_dBFS = target_dBFS - sound.dBFS
    sound = sound.apply_gain(change_dBFS)
    sound.export(target_path, 'wav')
    return jsonify(
        {"code": 0, "message": "转换成功", "source": filename + ".wav", "target": filename + "_output.wav", "txt": txt})


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/convert', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'static/uploads', str(int(time.time() * 1000)) + secure_filename(f.filename))
    f.save(upload_path)
    return do_service(upload_path)


@app.route('/reset')
def reset():
    pt = request.args['ckpt']
    return init(pt=pt, is_running=1)


@app.route('/ckpt')
def get_ckpt():
    return jsonify(
        {"code": 0, "ckpt": ckpt2})


@app.route('/ckpts')
def get_ckpts():
    models = sorted(glob.glob(logdir2 + "/" + "model-*.index"), key=os.path.getmtime, reverse=True)
    models = list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0], models))
    return jsonify({"code": 0, "list": models})


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    parser.add_argument('ip', type=str, help='to generate qrcode')
    parser.add_argument('port', type=int, help='port to run service')
    parser.add_argument('-ckpt', help='checkpoint to load models.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir2 = '{}/{}/train2'.format(hp.logdir_path, args.case2)
    import qrcode

    # 生成下载二维码
    qr = qrcode.QRCode(
        version=7,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4
    )
    qr.add_data("http://" + args.ip + ":" + str(args.port) + "/vc_demo.apk")
    qr.make(fit=True)
    img = qr.make_image()
    img.save("static/download.png")

    # 生成配置二维码
    qr = qrcode.QRCode(
        version=7,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4
    )
    qr.add_data("http://" + args.ip + ":" + str(args.port) + "/")
    qr.make(fit=True)
    img = qr.make_image()
    img.save("static/address.png")

    init(args)
    app.run(host='0.0.0.0', port=args.port, debug=True)
