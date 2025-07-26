[Japanese/[English](README_EN.md)]

# Audio-Processing-Node-Editor
ノードエディターベースのオーディオ処理アプリです。<br>
処理の検証や比較検討での用途を想定しています。<br>

<img src="https://github.com/user-attachments/assets/d59d10a4-8aef-4af4-af34-ef9ae6f682c7" loading="lazy" width="100%">

# Note
ノードは作成者(高橋)が必要になった順に追加しているため、<br>
オーディオ処理における基本的な処理を担うノードが不足していることがあります。<br>

# Requirements
```
dearpygui            2.0.0     or later
onnx                 1.17.0    or later
onnxruntime          1.17.0    or later
opencv-python        4.11.0.86 or later
librosa              0.11.0    or later
sounddevice          0.5.1     or later
soundfile            0.13.1    or later
webrtcvad-wheels     2.0.14    or later
vosk                 0.3.45    or later ※Speech Recognition(Vosk)ノードを実行する場合
google-cloud-speech  2.32.0    or later ※Speech Recognition(Google Speech-to-Text)ノードを実行する場合
```

Video Fileノードを使用する場合は、[FFmpeg](https://ffmpeg.org/) をインストールしてください。

# Installation
以下の何れかの方法で環境を準備してください。<br>
* スクリプトを直接実行
    1. リポジトリをクローン<br>`git clone https://github.com/Kazuhito00/Audio-Processing-Node-Editor`
    1. パッケージをインストール <br>`pip install -r requirements.txt`  
    1. 「main.py」を実行<br>`python main.py`
* 実行ファイルを利用(Windowsのみ)
    1. [apn-editor_win_x86_64.zip](https://github.com/Kazuhito00/Audio-Processing-Node-Editor/releases/download/v0.2.0/apn-editor_v0.2.0_win_x86_64.zip)をダウンロード
    1. 「main.exe」を実行 

# Usage
アプリの起動方法は以下です。
```bash
python main.py
```
* --setting<br>
ノードサイズやサンプリング周波数、Googleクレデンシャルパスの設定が記載された設定ファイルパスの指定<br>
デフォルト：node_editor/setting/setting.json

### Create Node
メニューから作成したいノードを選びクリック<br>
<img src="https://github.com/user-attachments/assets/4d9b810e-7e00-4084-b164-04412b093a60" loading="lazy" width="50%">

### Connect Node
出力端子をドラッグして入力端子に接続<br>
端子に設定された型同士のみ接続可能<br>
<img src="https://github.com/user-attachments/assets/47f26d31-1e78-4185-810b-7073ec53dd9d" loading="lazy" width="50%">

### Delete Node
削除したいノードを選択した状態で「Del」キー<br>
<img src="https://github.com/user-attachments/assets/d6f0e993-46f1-42a2-81a8-378ae0efc507" loading="lazy" width="50%">

### Export
メニューから「Export」を押し、ノード設定(jsonファイル)を保存<br>
<img src="https://github.com/user-attachments/assets/f3981cae-f58f-441e-b438-dee3ec83ed6d" loading="lazy" width="50%">

### Import
Exportで出力したノード設定(jsonファイル)を読み込む<br>
<img src="https://github.com/user-attachments/assets/3803a4a2-f60b-48a8-8ea0-3851bdfa1de8" loading="lazy" width="50%">

# Node
<details>
<summary>System Node</summary>

<table>
    <tr>
        <td width="200">
            Audio Control
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/610e47b5-3b18-4c77-a864-9efd04ef52ce" loading="lazy" width="300px">
        </td>
        <td width="760">
            Audio Fileノード、Micノード、Noiseノード、Write Wav Fileノード を制御するノード<br>
            Audio Controlノードはシステムで1つのみ生成可能
        </td>
    </tr>
</table>
</details>

<details>
<summary>Input Node</summary>

<table>
    <tr>
        <td width="200">
            Audio File
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/9fd91d47-e73a-461a-b3fc-eb4adac86e53" loading="lazy" width="300px">
        </td>
        <td width="760">
            オーディオファイル(wav, mp3, ogg, m4a)を読み込み、チャンクデータを出力するノード<br>
            「Select Audio File」ボタンでファイルダイアログをオープン
        </td>
    </tr>
    <tr>
        <td width="200">
            Mic
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/1e576e21-42cb-4788-a8aa-70fdf82452cd" loading="lazy" width="300px">
        </td>
        <td width="760">
            マイク入力を読み込み、チャンクデータを出力するノード<br>
            ドロップダウンリストからマイクを選択
        </td>
    </tr>
    <tr>
        <td width="200">
            Video File
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/26e095e3-5121-4d67-b3fe-3b81f9fd7a49" loading="lazy" width="300px">
        </td>
        <td width="760">
            動画ファイル(mp4, avi, webm)を読み込み、チャンクデータを出力するノード<br>
            「Select Audio File」ボタンでファイルダイアログをオープン<br>
            出力されるチャンクデータは全トラックを合成したデータ
        </td>
    </tr>
    <tr>
        <td width="200">
            Noise
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fd4d4dcf-a0ee-4e61-b07a-c7cc1bb8ecee" loading="lazy" width="300px">
        </td>
        <td width="760">
            ノイズを生成し、チャンクデータを出力するノード<br>
            ドロップダウンリストからノイズ種類（ホワイトノイズ、簡易ピンクノイズ、ヒスノイズ、ハムノイズ、パルスノイズ）を選択
        </td>
    </tr>
    <tr>
        <td width="200">
            Int Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031284-95255053-6eaf-4298-a392-062129e698f6.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            整数値を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Float Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031323-98ae0273-7083-48d0-9ef2-f02af7fde482.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            フロート値を出力するノード<br>
        </td>
    </tr>
</table>
</details>

<details>
<summary>Output Node</summary>

<table>
    <tr>
        <td width="200">
            Speaker
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/bd29d949-41a2-4d5a-b1d7-61667ffad61c" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、スピーカー出力を行うノード<br>
            ドロップダウンリストからスピーカーを選択
        </td>
    </tr>
    <tr>
        <td width="200">
            Write Wav File
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/0acf9e7d-3932-4ff6-a85e-3dcbdb38f406" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、Wavファイル保存を行うノード<br>
            出力先は「node_editor/setting/setting.json」の「output_directory」に設定<br>
            ※デフォルトは「./_output」
        </td>
    </tr>
</table>
</details>

<details>
<summary>Vol/Amp Node</summary>

<table>
    <tr>
        <td width="200">
            Gain Control(Scale Amplitude)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c46f5ff0-081a-4197-85e3-6b247d443573" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、定数倍したチャンクデータを出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Dynamic Range Compression
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/628484e8-3cb5-47dd-8e54-15428c8e23ee" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、ダイナミックレンジ圧縮を行ったチャンクデータを出力するノード<br>
            Threshold：閾値<br>
            Ratio：閾値を越えた値をどの程度の割合で圧縮するか<br>
            Attack(ms)：閾値を超えたとき、ゲインを下げる速さ（ミリ秒）<br>
            Release(ms)：閾値以下に戻ったとき、ゲインを戻す速さ（ミリ秒）<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Hard Limit
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/4a3c542c-fc7c-4a88-a96d-833ad02917ce" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、振幅制限を行ったチャンクデータを出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Soft Limit(tanh)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/ba18788f-4acc-4e9d-8d13-ea973a3e3825" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、tanhを用いた緩やかな振幅制限を行ったチャンクデータを出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Expander
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/3252d730-84de-4124-b142-9671c5ee730c" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、ダイナミックレンジ拡張を行ったチャンクデータを出力するノード<br>
            Threshold：閾値<br>
            Ratio：閾値を下回った値をどの程度の割合で減衰するか<br>
            Attack(ms)：閾値を超えたとき、ゲインを下げる速さ（ミリ秒）<br>
            Release(ms)：閾値以下に戻ったとき、ゲインを戻す速さ（ミリ秒）<br>
            Hold(ms)：閾値を下回ってもすぐに減衰しない猶予期間（ミリ秒）<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Noise Gate
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/697e99fe-862e-470c-b212-0b1194d0bea3" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、ノイズゲート処理を行ったチャンクデータを出力するノード<br>
            Threshold：閾値<br>
            Attack(ms)：閾値を超えたとき、ゲインを下げる速さ（ミリ秒）<br>
            Release(ms)：閾値以下に戻ったとき、ゲインを戻す速さ（ミリ秒）<br>
            Hold(ms)：閾値を下回ってもすぐに減衰しない猶予期間（ミリ秒）
        </td>
    </tr>
</table>
</details>

<details>
<summary>FreqDomain Node</summary>

<table>
    <tr>
        <td width="200">
            Bandpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fbd62390-5496-4e71-9e86-cca3fc33634d" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、バンドパスフィルター(バターワース IIR型)を通したチャンクデータを出力するノード<br>
            High Cut Freq(Hz)：上側遮断周波数(Hz)<br>
            Low Cut Freq(Hz)：下限遮断周波数(Hz)<br>
            Filter Order：フィルター次数
        </td>
    </tr>
    <tr>
        <td width="200">
            Bandstop Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a893e278-0b07-4e56-9877-d0413e4d5ad7" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、バンドストップフィルター(バターワース IIR型)を通したチャンクデータを出力するノード<br>
            High Cut Freq(Hz)：上側遮断周波数(Hz)<br>
            Low Cut Freq(Hz)：下限遮断周波数(Hz)<br>
            Filter Order：フィルター次数
        </td>
    </tr>
    <tr>
        <td width="200">
            Highpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c03f026d-baa2-4b47-b700-7a15eb565ee7" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、ハイパスフィルター(バターワース IIR型)を通したチャンクデータを出力するノード<br>
            Low Cut Freq(Hz)：下限遮断周波数(Hz)<br>
            Filter Order：フィルター次数
        </td>
    </tr>
    <tr>
        <td width="200">
            Lowpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/eae66eb8-2c1c-4050-8855-e094b31ba964" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、ローフィルター(バターワース IIR型)を通したチャンクデータを出力するノード<br>
            High Cut Freq(Hz)：上側遮断周波数(Hz)<br>
            Filter Order：フィルター次数
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple EQ(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/425bce4a-af9d-45d8-a3bb-f9953181e5c0" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、対象領域のゲイン増幅・減衰を行ったチャンクデータを出力するノード<br>
            High Cut Freq(Hz)：上側遮断周波数(Hz)<br>
            Low Cut Freq(Hz)：下限遮断周波数(Hz)<br>
            Gain(dB)：ゲイン(dB)
        </td>
    </tr>
    <tr>
        <td width="200">
            Power Spectrum
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a6b68760-eeb4-4887-9d3b-6354b8d91fbd" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、簡易的なパワースペクトルを表示するノード<br>
            -80dB～40dBの表示範囲
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple Spectrogram
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/2d9b8e18-1348-40ee-93d6-26f109fdeb82" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、簡易的なスペクトログラムを表示するノード<br>
            シフトサイズや窓関数(ハミング窓かハニング窓)、平滑化数などは「node_editor/setting/setting.json」に設定<br>
            表示データの下端が0Hz、上端がナイキスト周波数
        </td>
    </tr>
</table>
</details>

<details>
<summary>TimeDomain Node</summary>

<table>
    <tr>
        <td width="200">
            Start Delay
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/0a7a661e-0f9b-441c-af6c-70068162e6dc" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、指定時間遅延したチャンクデータを出力するノード<br>
            ※時間指定は、Audio Controlノードで「停止」中に行うこと
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple Mixer
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c8ed526a-404b-4ccb-aa1a-5a307a7dd00b" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータ2つを受け取り、ミキシングしたチャンクデータを出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Voice Activity Detection(Silero VAD)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/7df59ac0-2ee6-454b-93b5-31cb6127b262" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、Silero VADを用いた音声区間検出を行うノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Voice Activity Detection(WebRTC VAD)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/372418e7-61b7-4f08-8606-e649535e68e5"" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、WebRTC VADを用いた音声区間検出を行うノード<br>
            aggressive は VAD の感度指定で、0がデフォルト、3が最高感度です。
        </td>
    </tr>
</table>
</details>

<details>
<summary>AudioEnhance Node</summary>

<table>
    <tr>
        <td width="200">
            Speech Enhancement(GTCRN)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/cecd2dc2-a673-4547-8549-4dabead84ee7" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、GTCRNを用いた音声強調を行ったチャンクデータを出力するノード
        </td>
    </tr>
</table>
</details>

<details>
<summary>Analysis Node</summary>

<table>
    <tr>
        <td width="200">
            Speech Recognition(Google Speech-to-Text)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/d8aa01fe-b09c-449f-b30e-82e140c67d90" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、Google Speech-to-Textを用いて文字書き起こしを行うノード<br>
            現在は「日本語」と「English」のみ対応。<br>
            このノードを使用する際は、「node_editor/setting/setting.json」の「google_application_credentials_json」にサービスアカウントキーを設定してください<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Speech Recognition(Vosk)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/e4deca35-57f7-47bf-95ab-30c64e9c2719" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータを受け取り、Voskを用いて文字書き起こしを行うノード<br>
            現在は「日本語」と「English」のみ対応。
        </td>
    </tr>
</table>
</details>

<details>
<summary>Other Node</summary>

<table>
    <tr>
        <td width="200">
            Switch
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fd2a8bb7-ed7f-40c0-a632-eb6b5d77f22d" loading="lazy" width="300px">
        </td>
        <td width="760">
            チャンクデータ2つを受け取り、指定したチャンクデータを出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            FPS
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a9e28170-98eb-4b72-b0ce-da23e7d05b21" loading="lazy" width="300px">
        </td>
        <td width="760">
            処理時間を計測するノード
        </td>
    </tr>
</table>
</details>

# Node(Another repository)
他リポジトリで公開しているノードです。<br>
Audio-Processing-Node-Editor で使用するには、各リポジトリのインストール方法に従ってください。

<details>
<summary>Input Node</summary>

<table>
    <tr>
        <td width="200">
            <a href=https://github.com/Kazuhito00/APNE-Input-getUserMedia-Mic-Node>Mic(getUserMedia())</a> 
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a5231445-5c0c-4f43-92dd-c1e1d913c108" loading="lazy" width="300px">
        </td>
        <td width="760">
            WebブラウザのgetUserMedia()経由で取得したマイク入力を扱うノード<br>
            ノードを生成するとブラウザが立ち上がります。<br>
          　「1. Prepare Microphone」を押下後、マイク使用を許可し、「Start Recording」を押下してください。
        </td>
    </tr>
</table>

</details>

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Audio-Processing-Node-Editor is under [Apache-2.0 license](LICENSE).<br><br>
Audio-Processing-Node-Editorのソースコード自体は[Apache-2.0 license](LICENSE)ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。<br>
詳細は各ディレクトリ同梱のLICENSEファイルをご確認ください。
